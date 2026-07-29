"""
Image generation module — GPT Image 2 (OpenAI).

Exposes:
  - generate_image(...)          the raw generation function
  - router                       FastAPI router with:
                                   POST /api/image/generate
                                   GET  /api/image/status/{job_id}

Required environment variable:
  OPENAI_API_KEY
"""

import base64
import logging
import os
import tempfile
import threading
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from job_runtime import create_job, get_job, submit_job, update_job
from s3_storage import normalize_user_id, store_generated_media

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

_ALLOWED_IMAGE_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}
_ALLOWED_QUALITY = {"low", "medium", "high", "auto"}

router = APIRouter(prefix="/api/image", tags=["image"])
logger = logging.getLogger(__name__)

_openai_client = None
_openai_client_lock = threading.Lock()


def _get_openai_client():
    """Return a cached OpenAI client (one per process)."""
    global _openai_client
    from openai import OpenAI

    if _openai_client is not None:
        return _openai_client

    with _openai_client_lock:
        if _openai_client is not None:
            return _openai_client

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Server misconfigured: OPENAI_API_KEY environment variable is not set."
            )
        # Image generation can take 30–120s; keep a long read timeout.
        _openai_client = OpenAI(api_key=api_key, timeout=180.0, max_retries=2)
        return _openai_client


async def _optional_reference_bytes(upload: Optional[UploadFile]) -> Optional[tuple[bytes, str]]:
    """
    Returns (bytes, suffix) for a real image upload, else None.

    Treats Swagger/docs empty sends as missing:
      - no field
      - empty filename
      - zero-byte body
      - filename placeholders like "string"
    """
    if upload is None:
        return None

    name = (upload.filename or "").strip()
    if not name or name.lower() in {"string", "null", "undefined", "blob"}:
        # Still allow a real blob upload with no filename if body has bytes
        content = await upload.read()
        if not content:
            return None
        content_type = (upload.content_type or "").lower()
        if content_type.startswith("image/"):
            ext = ".png" if "png" in content_type else ".jpg"
            return content, ext
        return None

    content = await upload.read()
    if not content:
        return None

    suffix = os.path.splitext(name)[1] or ".png"
    return content, suffix


def generate_image(
    prompt: str,
    reference_image_paths: Optional[list] = None,
    size: str = "1536x1024",
    quality: str = "medium",
    output_path: str = "output.png",
) -> str:
    """Calls GPT Image 2 and writes the resulting PNG to output_path."""
    client = _get_openai_client()

    valid_refs = [
        p for p in (reference_image_paths or [])
        if p and os.path.isfile(p) and os.path.getsize(p) > 0
    ]

    if valid_refs:
        image_files = [open(p, "rb") for p in valid_refs]
        try:
            result = client.images.edit(
                model="gpt-image-2",
                image=image_files,
                prompt=prompt,
                size=size,
                quality=quality,
            )
        finally:
            for f in image_files:
                f.close()
    else:
        # Prompt-only generation (no reference image)
        result = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size=size,
            quality=quality,
        )

    if not result.data or not result.data[0].b64_json:
        raise RuntimeError("OpenAI returned no image data")

    image_base64 = result.data[0].b64_json
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    return output_path


def _run_image_job(
    job_id: str,
    prompt: str,
    size: str,
    quality: str,
    user_id: str,
    reference_image_path: Optional[str] = None,
):
    """Background worker: generate image → upload S3 → update job status."""
    output_path = os.path.join(GENERATED_DIR, f"{job_id}.png")
    try:
        update_job(job_id, status="processing", progress="Generating image")

        generate_image(
            prompt=prompt,
            reference_image_paths=[reference_image_path] if reference_image_path else None,
            size=size,
            quality=quality,
            output_path=output_path,
        )

        update_job(job_id, progress="Uploading to S3")
        public_url, s3_key = store_generated_media(
            local_path=output_path,
            user_id=user_id,
            media_kind="images",
            filename=f"{job_id}.png",
            content_type="image/png",
            delete_local=True,
        )

        update_job(
            job_id,
            status="completed",
            progress="Done",
            url=public_url,
            s3_key=s3_key,
            filename=f"{job_id}.png",
        )
    except Exception as exc:
        logger.exception("Image job %s failed", job_id)
        update_job(job_id, status="failed", error=str(exc), progress=None)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
    finally:
        if reference_image_path and os.path.exists(reference_image_path):
            try:
                os.remove(reference_image_path)
            except OSError:
                pass


class ImageGenerateResponse(BaseModel):
    job_id: str
    status: str
    user_id: str


class ImageStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    progress: Optional[str] = None
    url: Optional[str] = None
    filename: Optional[str] = None
    s3_key: Optional[str] = None
    user_id: Optional[str] = None
    prompt: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    error: Optional[str] = None


@router.post("/generate", response_model=ImageGenerateResponse)
async def api_generate_image(
    prompt: str = Form(..., description="Image prompt (required)"),
    user_id: str = Form(..., description="Logged-in AdvPost user id"),
    size: str = Form("1536x1024"),
    quality: str = Form("medium"),
    reference_image: Annotated[
        Optional[UploadFile],
        File(description="Optional reference image. Leave empty for prompt-only generation."),
    ] = None,
):
    """
    Starts an async image generation job on a worker thread.

    GPT Image can take 30–120s, which exceeds Cloudflare's proxy timeout when
    done synchronously. Returns a job_id immediately — poll
    GET /api/image/status/{job_id} until status is completed or failed.

    multipart/form-data fields:
      - prompt (required)
      - user_id (required): logged-in AdvPost user id
      - size: one of 1024x1024, 1536x1024, 1024x1536, auto (default 1536x1024)
      - quality: one of low, medium, high, auto (default medium)
      - reference_image: OPTIONAL — omit entirely for text-to-image
    """
    if size not in _ALLOWED_IMAGE_SIZES:
        raise HTTPException(400, f"size must be one of {sorted(_ALLOWED_IMAGE_SIZES)}")
    if quality not in _ALLOWED_QUALITY:
        raise HTTPException(400, f"quality must be one of {sorted(_ALLOWED_QUALITY)}")

    try:
        uid = normalize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    tmp_ref_path = None

    ref = await _optional_reference_bytes(reference_image)
    if ref is not None:
        content, suffix = ref
        tmp_ref_path = os.path.join(tempfile.gettempdir(), f"ref_{job_id}{suffix}")
        with open(tmp_ref_path, "wb") as f:
            f.write(content)

    create_job(
        job_id,
        status="queued",
        progress=None,
        url=None,
        filename=None,
        s3_key=None,
        user_id=uid,
        prompt=prompt,
        size=size,
        quality=quality,
        error=None,
    )

    submit_job(
        _run_image_job,
        job_id,
        prompt,
        size,
        quality,
        uid,
        tmp_ref_path,
    )

    return ImageGenerateResponse(job_id=job_id, status="queued", user_id=uid)


@router.get("/status/{job_id}", response_model=ImageStatusResponse)
async def api_image_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "job_id not found")
    # Drop internal timestamps from the API response model
    return ImageStatusResponse(
        job_id=job["job_id"],
        status=job.get("status", "queued"),
        progress=job.get("progress"),
        url=job.get("url"),
        filename=job.get("filename"),
        s3_key=job.get("s3_key"),
        user_id=job.get("user_id"),
        prompt=job.get("prompt"),
        size=job.get("size"),
        quality=job.get("quality"),
        error=job.get("error"),
    )

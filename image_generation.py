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
from pydantic import BaseModel, BeforeValidator, WithJsonSchema

from job_runtime import create_job, get_job, submit_job, update_job
from s3_storage import normalize_user_id, store_generated_media

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

_ALLOWED_IMAGE_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}
_ALLOWED_QUALITY = {"low", "medium", "high", "auto"}
_MAX_LANGUAGE_LEN = 64

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
    reference_image_paths: Optional[list] = None,
):
    """Background worker: generate image → upload S3 → update job status."""
    output_path = os.path.join(GENERATED_DIR, f"{job_id}.png")
    reference_image_paths = reference_image_paths or []
    try:
        update_job(job_id, status="processing", progress="Generating image")

        generate_image(
            prompt=prompt,
            reference_image_paths=reference_image_paths,
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
        for ref_path in reference_image_paths:
            if ref_path and os.path.exists(ref_path):
                try:
                    os.remove(ref_path)
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
    language: Optional[str] = None
    error: Optional[str] = None


_MAX_PRODUCT_IMAGES = 4

# Swagger UI only renders "Choose file" for format:binary. FastAPI 0.129+ emits
# OAS 3.1 contentMediaType for UploadFile, which makes list[UploadFile] show as
# text inputs ("Add string item") instead of file pickers.


def _coerce_optional_upload(value):
    """Treat Swagger/curl empty file fields ('') as missing."""
    if value is None or value == "" or isinstance(value, str):
        return None
    return value


def _coerce_upload_list(value):
    """
    Swagger 'Send empty value' / curl -F 'product_images=' sends '' which
    becomes ['']. Drop non-file placeholders so validation does not 422.
    """
    if value is None or value == "":
        return []
    if not isinstance(value, list):
        value = [value]
    return [
        item
        for item in value
        if item is not None and not isinstance(item, str) and item != ""
    ]


_OptionalUpload = Annotated[
    Optional[UploadFile],
    BeforeValidator(_coerce_optional_upload),
    WithJsonSchema({"type": "string", "format": "binary"}),
]

_ProductUploadList = Annotated[
    list[UploadFile],
    BeforeValidator(_coerce_upload_list),
    WithJsonSchema(
        {
            "type": "array",
            "items": {"type": "string", "format": "binary"},
        }
    ),
]


def _palette_prompt_block(color_palette: str) -> str:
    """Build a strict palette instruction from a comma-separated hex list."""
    colors = [c.strip() for c in color_palette.split(",") if c.strip()]
    if not colors:
        return ""
    return (
        "\n\nCOLOR PALETTE (strict):\n"
        f"Use exactly this color palette for the poster: {', '.join(colors)}. "
        "Use the first color as the dominant/brand color and the rest as "
        "supporting/background/accent colors. Do not introduce clashing colors."
    )


def _language_prompt_block(language: str) -> str:
    """Build a strict on-image text language instruction for any language."""
    return (
        f"\n\nLANGUAGE (strict — highest priority):\n"
        f"Target language: {language}.\n"
        f"- Every piece of readable text in the image (headlines, slogans, offers, "
        f"CTAs, labels, captions, prices, dates, and body copy) MUST be written "
        f"only in {language}, using the correct native script for that language.\n"
        f"- If the prompt or product copy is in another language, translate it "
        f"naturally into {language} before rendering it on the image.\n"
        f"- Do not mix languages. Do not leave leftover English (or any other "
        f"language) text on the poster unless the target language itself is that "
        f"language.\n"
        f"- Brand names and logos may stay as provided / as shown in reference "
        f"images; everything else must be {language} only."
    )


def _reference_assets_prompt_block(
    *,
    has_logo: bool,
    product_count: int,
    has_legacy_ref: bool,
) -> str:
    """
    Describe attached reference files and force flexible product placement.

    Without this, images.edit often pastes the product stuck on the left.
    """
    if not has_logo and product_count <= 0 and not has_legacy_ref:
        return ""

    lines = [
        "\n\nATTACHED REFERENCE IMAGES (in order):",
    ]
    idx = 1
    if has_logo:
        lines.append(
            f"- Image {idx}: BUSINESS LOGO — place as a small brand mark "
            f"(typically corner). Keep the logo recognizable; do not redraw "
            f"or invent a different logo."
        )
        idx += 1
    for i in range(product_count):
        lines.append(
            f"- Image {idx}: PRODUCT PHOTO {i + 1} — use the real product "
            f"appearance as a hero visual. Keep product shape, packaging, "
            f"colors, and labels faithful to the photo."
        )
        idx += 1
    if has_legacy_ref:
        lines.append(
            f"- Image {idx}: STYLE/REFERENCE image — use only for mood, "
            f"palette, or composition inspiration."
        )

    lines.append(
        "\nPRODUCT PLACEMENT (strict — do NOT default to left side):\n"
        "- Freely plan the layout. Place product(s) wherever best fits a strong "
        "ad composition: center, right, bottom, top, foreground, slight angle, "
        "or integrated into the scene — NOT locked to the left half.\n"
        "- Vary position and scale across designs. Do not paste the product "
        "as a fixed left-column cutout or mirrored collage.\n"
        "- Balance text and product so neither is cramped. Leave clear space "
        "for headlines/CTAs opposite or around the product.\n"
        "- Composite the product naturally into the poster (lighting, shadow, "
        "perspective) so it looks designed-in, not stuck on."
    )
    return "\n".join(lines)


@router.post("/generate", response_model=ImageGenerateResponse)
async def api_generate_image(
    prompt: str = Form(..., description="Image prompt (required)"),
    user_id: str = Form(..., description="Logged-in AdvPost user id"),
    size: str = Form("1536x1024"),
    quality: str = Form("medium"),
    language: str = Form(
        "Marathi",
        description=(
            "Language for all on-image text (any language name, e.g. English, "
            "Hindi, Marathi, Tamil, Spanish). Default: Marathi"
        ),
    ),
    color_palette: str = Form(
        "",
        description="Optional comma-separated hex colors, e.g. #7C3AED,#C4B5FD,#1E1B4B",
    ),
    logo_image: Annotated[
        _OptionalUpload,
        File(description="Optional business logo (placed as brand mark, never redrawn)."),
    ] = None,
    product_images: Annotated[
        _ProductUploadList,
        File(description="Optional product photos (up to 4) used as hero visuals."),
    ] = [],
    reference_image: Annotated[
        _OptionalUpload,
        File(description="Optional generic reference image (legacy field)."),
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
      - language: any language name for on-image text (default Marathi)
      - color_palette: OPTIONAL comma-separated hex colors for the poster
      - logo_image: OPTIONAL business logo file
      - product_images: OPTIONAL product photos (repeat field, up to 4)
      - reference_image: OPTIONAL legacy generic reference image
    """
    if size not in _ALLOWED_IMAGE_SIZES:
        raise HTTPException(400, f"size must be one of {sorted(_ALLOWED_IMAGE_SIZES)}")
    if quality not in _ALLOWED_QUALITY:
        raise HTTPException(400, f"quality must be one of {sorted(_ALLOWED_QUALITY)}")
    language = language.strip()
    if not language or len(language) > _MAX_LANGUAGE_LEN:
        raise HTTPException(
            400,
            f"language must be a non-empty name up to {_MAX_LANGUAGE_LEN} characters",
        )

    try:
        uid = normalize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex

    # Attachment order matters and is described to the model in the prompt:
    # logo first, then product photos, then any legacy reference image.
    ref_paths: list[str] = []
    has_logo = False
    product_count = 0
    has_legacy_ref = False

    async def _save_upload(upload: Optional[UploadFile], tag: str) -> bool:
        data = await _optional_reference_bytes(upload)
        if data is None:
            return False
        content, suffix = data
        path = os.path.join(
            tempfile.gettempdir(), f"{tag}_{job_id}_{len(ref_paths)}{suffix}"
        )
        with open(path, "wb") as f:
            f.write(content)
        ref_paths.append(path)
        return True

    if await _save_upload(logo_image, "logo"):
        has_logo = True
    for product in product_images[:_MAX_PRODUCT_IMAGES]:
        if await _save_upload(product, "product"):
            product_count += 1
    if await _save_upload(reference_image, "ref"):
        has_legacy_ref = True

    # Language + layout first so the model treats them as hard constraints.
    prompt_parts = [_language_prompt_block(language).strip()]
    assets_block = _reference_assets_prompt_block(
        has_logo=has_logo,
        product_count=product_count,
        has_legacy_ref=has_legacy_ref,
    ).strip()
    if assets_block:
        prompt_parts.append(assets_block)
    prompt_parts.append(prompt.strip())
    final_prompt = "\n\n".join(prompt_parts)
    if color_palette.strip():
        final_prompt = f"{final_prompt}{_palette_prompt_block(color_palette)}"

    create_job(
        job_id,
        status="queued",
        progress=None,
        url=None,
        filename=None,
        s3_key=None,
        user_id=uid,
        prompt=final_prompt,
        size=size,
        quality=quality,
        language=language,
        error=None,
    )

    submit_job(
        _run_image_job,
        job_id,
        final_prompt,
        size,
        quality,
        uid,
        ref_paths,
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
        language=job.get("language"),
        error=job.get("error"),
    )

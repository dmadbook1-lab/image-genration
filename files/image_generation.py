"""
Image generation module — GPT Image 2 (OpenAI).

Exposes:
  - generate_image(...)          the raw generation function
  - router                       FastAPI router with POST /api/image/generate

Required environment variable:
  OPENAI_API_KEY
"""

import base64
import os
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

_ALLOWED_IMAGE_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}
_ALLOWED_QUALITY = {"low", "medium", "high", "auto"}

router = APIRouter(prefix="/api/image", tags=["image"])


def _public_url(filename: str) -> str:
    return f"/files/{filename}"


def _get_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: OPENAI_API_KEY environment variable is not set.",
        )
    return OpenAI(api_key=api_key)


def generate_image(
    prompt: str,
    reference_image_paths: Optional[list] = None,
    size: str = "1536x1024",
    quality: str = "high",
    output_path: str = "output.png",
) -> str:
    """Calls GPT Image 2 and writes the resulting PNG to output_path."""
    client = _get_openai_client()

    if reference_image_paths:
        image_files = [open(p, "rb") for p in reference_image_paths]
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
        result = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size=size,
            quality=quality,
        )

    image_base64 = result.data[0].b64_json
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    return output_path


class ImageGenerateResponse(BaseModel):
    id: str
    filename: str
    url: str
    prompt: str
    size: str
    quality: str


@router.post("/generate", response_model=ImageGenerateResponse)
async def api_generate_image(
    prompt: str = Form(...),
    size: str = Form("1536x1024"),
    quality: str = Form("high"),
    reference_image: Optional[UploadFile] = File(None),
):
    """
    Synchronous image generation. Returns immediately with a downloadable URL.

    multipart/form-data fields:
      - prompt (required)
      - size: one of 1024x1024, 1536x1024, 1024x1536, auto (default 1536x1024)
      - quality: one of low, medium, high, auto (default high)
      - reference_image: optional image file to edit/use as reference
    """
    if size not in _ALLOWED_IMAGE_SIZES:
        raise HTTPException(400, f"size must be one of {sorted(_ALLOWED_IMAGE_SIZES)}")
    if quality not in _ALLOWED_QUALITY:
        raise HTTPException(400, f"quality must be one of {sorted(_ALLOWED_QUALITY)}")

    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    output_path = os.path.join(GENERATED_DIR, filename)

    reference_paths = None
    tmp_ref_path = None
    try:
        if reference_image is not None:
            suffix = os.path.splitext(reference_image.filename or "")[1] or ".png"
            tmp_ref_path = os.path.join(tempfile.gettempdir(), f"ref_{image_id}{suffix}")
            with open(tmp_ref_path, "wb") as f:
                f.write(await reference_image.read())
            reference_paths = [tmp_ref_path]

        generate_image(
            prompt=prompt,
            reference_image_paths=reference_paths,
            size=size,
            quality=quality,
            output_path=output_path,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Image generation failed: {exc}")
    finally:
        if tmp_ref_path and os.path.exists(tmp_ref_path):
            os.remove(tmp_ref_path)

    return ImageGenerateResponse(
        id=image_id,
        filename=filename,
        url=_public_url(filename),
        prompt=prompt,
        size=size,
        quality=quality,
    )

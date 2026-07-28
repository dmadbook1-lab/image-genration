"""
Upload generated media to AWS S3 (bucket: advpost).

Canonical layout (same as PHP media_helper):
  users/{user_id}/generated/images/{filename}
  users/{user_id}/generated/videos/{filename}
"""

from __future__ import annotations

import mimetypes
import os
import re
from typing import Optional

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def s3_enabled() -> bool:
    return bool(
        os.environ.get("AWS_ACCESS_KEY_ID")
        and os.environ.get("AWS_SECRET_ACCESS_KEY")
        and os.environ.get("AWS_BUCKET_NAME")
    )


def s3_required() -> bool:
    """When true (default), generation fails if S3 is not configured."""
    raw = (os.environ.get("S3_REQUIRED") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def require_s3() -> None:
    if not s3_enabled():
        raise RuntimeError(
            "AWS S3 is not configured. Set AWS_ACCESS_KEY_ID, "
            "AWS_SECRET_ACCESS_KEY, and AWS_BUCKET_NAME in .env "
            "(same credentials as the PHP aws_credential table)."
        )


def normalize_user_id(user_id: Optional[str]) -> str:
    """Sanitize user id for S3 path safety. Raises ValueError if invalid."""
    uid = (user_id or "").strip()
    if not uid or uid.lower() in {"anonymous", "null", "undefined", "0"}:
        raise ValueError("user_id is required so media can be stored under that user folder")
    if not _USER_ID_RE.match(uid):
        raise ValueError("user_id must be alphanumeric (optionally with _ or -)")
    return uid


def safe_filename(name: str) -> str:
    base = os.path.basename((name or "").strip()) or "file.bin"
    base = re.sub(r"[^A-Za-z0-9_\-.]", "_", base)
    return base or "file.bin"


def user_generated_key(user_id: str, media_kind: str, filename: str) -> str:
    """
    media_kind: 'images' | 'videos'
    → users/{user_id}/generated/{media_kind}/{filename}
    """
    uid = normalize_user_id(user_id)
    kind = (media_kind or "").strip().lower()
    if kind not in {"images", "videos"}:
        raise ValueError("media_kind must be 'images' or 'videos'")
    return f"users/{uid}/generated/{kind}/{safe_filename(filename)}"


def s3_public_url(key: str) -> str:
    base = os.environ.get("AWS_URL", "").rstrip("/")
    if not base:
        bucket = os.environ["AWS_BUCKET_NAME"]
        region = os.environ.get("AWS_REGION", "ap-south-1")
        base = f"https://{bucket}.s3.{region}.amazonaws.com"
    return f"{base}/{key.lstrip('/')}"


def upload_file_to_s3(local_path: str, key: str, content_type: Optional[str] = None) -> str:
    """Upload a local file and return the public S3 URL."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    require_s3()

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Local file not found for S3 upload: {local_path}")

    bucket = os.environ["AWS_BUCKET_NAME"]
    region = os.environ.get("AWS_REGION", "ap-south-1")
    key = key.lstrip("/").replace("\\", "/")

    if content_type is None:
        content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"

    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    extra = {
        "ContentType": content_type,
        "CacheControl": "public, max-age=31536000",
    }

    try:
        client.upload_file(local_path, bucket, key, ExtraArgs=extra)
        # Confirm object exists
        client.head_object(Bucket=bucket, Key=key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed for s3://{bucket}/{key}: {exc}") from exc

    return s3_public_url(key)


def store_generated_media(
    local_path: str,
    user_id: str,
    media_kind: str,
    filename: str,
    content_type: Optional[str] = None,
    delete_local: bool = False,
) -> tuple[str, str]:
    """
    Upload generated media under the user's S3 folder.
    Returns (public_url, s3_key).
    """
    if s3_required() or s3_enabled():
        require_s3()
        key = user_generated_key(user_id, media_kind, filename)
        url = upload_file_to_s3(local_path, key, content_type)
        if delete_local:
            try:
                os.remove(local_path)
            except OSError:
                pass
        return url, key

    # Local-only fallback when S3_REQUIRED=0 and AWS not configured
    return f"/files/{safe_filename(filename)}", ""

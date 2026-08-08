"""
List previously generated media for a user from S3.

GET /api/media/generated?user_id=...&kind=all|images|videos
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from s3_storage import list_user_generated_media, normalize_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/media", tags=["media"])


class GeneratedMediaItem(BaseModel):
    id: str
    kind: Literal["image", "video"]
    filename: str
    url: str
    s3_key: str
    size_bytes: int | None = None
    last_modified: str | None = None


class GeneratedMediaListResponse(BaseModel):
    user_id: str
    items: list[GeneratedMediaItem]


@router.get("/generated", response_model=GeneratedMediaListResponse)
async def list_generated_media(
    user_id: str = Query(..., description="Logged-in AdvPost user id"),
    kind: str = Query("all", description="all | images | videos"),
):
    """Return the user's AI-generated posters and reels still on S3."""
    try:
        uid = normalize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    kind_norm = (kind or "all").strip().lower()
    if kind_norm not in {"all", "images", "videos", "image", "video", "poster", "reel"}:
        raise HTTPException(400, "kind must be all, images, or videos")

    try:
        raw: list[dict] = []
        if kind_norm in {"all", "images", "image", "poster"}:
            raw.extend(list_user_generated_media(uid, "images"))
        if kind_norm in {"all", "videos", "video", "reel"}:
            raw.extend(list_user_generated_media(uid, "videos"))
    except Exception as exc:
        logger.exception("Failed listing generated media for user %s", uid)
        raise HTTPException(500, f"Failed to list generated media: {exc}") from exc

    items: list[GeneratedMediaItem] = []
    for row in raw:
        filename = row["filename"]
        key = row["s3_key"]
        media_kind: Literal["image", "video"] = (
            "video" if "/videos/" in key else "image"
        )
        items.append(
            GeneratedMediaItem(
                id=filename.rsplit(".", 1)[0],
                kind=media_kind,
                filename=filename,
                url=row["url"],
                s3_key=key,
                size_bytes=row.get("size_bytes"),
                last_modified=row.get("last_modified"),
            )
        )

    items.sort(key=lambda i: i.last_modified or "", reverse=True)
    return GeneratedMediaListResponse(user_id=uid, items=items)

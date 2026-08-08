"""
Fire-and-forget generation log shipping to AdvPost PHP admin backend.

Env:
  ADVPOST_API_BASE_URL  e.g. http://127.0.0.1:8080/index.php/api/PostController/
  MEDIA_LOG_SECRET      shared with PHP MEDIA_LOG_SECRET / X-Media-Log-Key
  MEDIA_LOG_ENABLED     1/0 (default 1 when base URL is set)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    raw = (os.environ.get("MEDIA_LOG_ENABLED") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    base = (os.environ.get("ADVPOST_API_BASE_URL") or "").strip()
    return bool(base)


def _endpoint() -> str:
    base = (os.environ.get("ADVPOST_API_BASE_URL") or "").rstrip("/")
    if base.endswith("PostController"):
        return f"{base}/log_generation"
    if base.endswith("PostController/"):
        return f"{base}log_generation"
    return f"{base}/log_generation"


def _post_sync(payload: dict[str, Any]) -> None:
    try:
        import httpx
    except ModuleNotFoundError:
        logger.warning("httpx missing — cannot ship generation log")
        return

    secret = (os.environ.get("MEDIA_LOG_SECRET") or "advpost-media-log-2026").strip()
    url = _endpoint()
    try:
        resp = httpx.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Media-Log-Key": secret,
            },
            timeout=8.0,
        )
        if resp.status_code >= 400:
            logger.warning(
                "generation log POST failed %s: %s",
                resp.status_code,
                (resp.text or "")[:300],
            )
        else:
            logger.info("generation log shipped job_id=%s status=%s", payload.get("job_id"), payload.get("status"))
    except Exception:
        logger.exception("generation log POST error for job_id=%s", payload.get("job_id"))


def log_generation_event(payload: dict[str, Any], *, background: bool = True) -> None:
    """
    Upsert a generation log on the PHP backend.
    Never raises — logging must not break generation.
    """
    if not _enabled():
        return

    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        clean[key] = value

    if "job_id" not in clean or "user_id" not in clean or "media_kind" not in clean:
        logger.warning("generation log skipped — missing required fields")
        return

    # Ensure user_id is int-compatible string/int
    try:
        clean["user_id"] = int(str(clean["user_id"]).strip())
    except (TypeError, ValueError):
        logger.warning("generation log skipped — invalid user_id=%r", clean.get("user_id"))
        return

    if background:
        threading.Thread(target=_post_sync, args=(clean,), daemon=True).start()
    else:
        _post_sync(clean)


def dumps_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)

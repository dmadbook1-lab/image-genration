"""
Shared runtime for media generation jobs.

- ThreadPoolExecutor so sync OpenAI/Veo/S3 work does not block the FastAPI event loop
- In-memory job store helpers with TTL pruning for completed/failed jobs
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_JOB_TTL_SECONDS = int(os.environ.get("MEDIA_JOB_TTL_SECONDS", str(60 * 60)))  # 1 hour
_MAX_WORKERS = max(1, int(os.environ.get("MEDIA_MAX_WORKERS", "3")))

_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

# Shared store used by image + video routers
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_MAX_WORKERS,
                thread_name_prefix="media-job",
            )
            logger.info("Media job executor started (max_workers=%s)", _MAX_WORKERS)
        return _executor


def submit_job(fn: Callable[..., Any], *args: Any, **kwargs: Any):
    """Run a sync job function on the shared thread pool."""
    return get_executor().submit(fn, *args, **kwargs)


def create_job(job_id: str, **fields: Any) -> dict[str, Any]:
    """Insert a new job record. Adds created_at / updated_at timestamps."""
    now = time.time()
    record = {
        "job_id": job_id,
        "created_at": now,
        "updated_at": now,
        **fields,
    }
    with _jobs_lock:
        _prune_locked(now)
        _jobs[job_id] = record
        return dict(record)


def update_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(fields)
        job["updated_at"] = time.time()


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _prune_locked(now: Optional[float] = None) -> None:
    """Drop completed/failed jobs older than TTL. Caller must hold _jobs_lock."""
    now = now if now is not None else time.time()
    stale = [
        jid
        for jid, job in _jobs.items()
        if job.get("status") in {"completed", "failed"}
        and (now - float(job.get("updated_at") or job.get("created_at") or now)) > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        _jobs.pop(jid, None)
    if stale:
        logger.debug("Pruned %s expired media jobs", len(stale))

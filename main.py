"""
Main FastAPI entrypoint. Wires together the image and video routers.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Required environment variables:
    OPENAI_API_KEY                 - your OpenAI key (DO NOT hardcode it)
    GOOGLE_CLOUD_PROJECT            - GCP project for Veo/Gemini (defaults to SA JSON project_id)
    GOOGLE_CLOUD_REGION             - defaults to "us-central1" if unset
    (optional) GOOGLE_APPLICATION_CREDENTIALS - override path to SA JSON;
        default: advpost-video-27ad9b6f9358.json beside this service

Optional tuning:
    MEDIA_MAX_WORKERS               - concurrent image/video jobs (default 3)
    MEDIA_JOB_TTL_SECONDS           - prune completed/failed jobs after N seconds (default 3600)
    FFMPEG_BINARY                   - ffmpeg to run (default "ffmpeg" on PATH); must
                                      include the drawtext filter for text overlays

Also requires the `ffmpeg` binary (with drawtext/libfreetype) on the host machine.
See README.md for full endpoint documentation.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

from image_generation import router as image_router
from media_library import router as media_library_router
from video_generation import router as video_router

app = FastAPI(title="Media Generation API", version="1.0.0")

# Allow your Flutter app (web/mobile/desktop) to call this API during dev.
# Lock this down to your actual domains before shipping to production.

# Allow only requests from the specified origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# Serves generated files at http://<host>:8000/files/<filename>
app.mount("/files", StaticFiles(directory=GENERATED_DIR), name="files")

app.include_router(image_router)
app.include_router(video_router)
app.include_router(media_library_router)


@app.on_event("startup")
async def on_startup():
    # Warm the shared executor so first job doesn't pay setup cost on request path.
    from job_runtime import get_executor
    from video_generation import FFMPEG_BIN, drawtext_available, ffmpeg_available

    get_executor()
    if not ffmpeg_available():
        logger.error(
            "ffmpeg binary %r not found — video jobs will fail. "
            "Install ffmpeg on this host (e.g. apt-get install ffmpeg) "
            "or set FFMPEG_BINARY.",
            FFMPEG_BIN,
        )
    logger.info(
        "Media Generation API ready (MEDIA_MAX_WORKERS=%s, ffmpeg=%s, drawtext=%s)",
        os.environ.get("MEDIA_MAX_WORKERS", "3"),
        "ok" if ffmpeg_available() else "MISSING",
        "ok" if drawtext_available() else "MISSING (videos will have no text overlay)",
    )


@app.get("/health")
async def health():
    from video_generation import drawtext_available, ffmpeg_available

    return {
        "status": "ok",
        "ffmpeg": "ok" if ffmpeg_available() else "missing",
        "drawtext": "ok" if drawtext_available() else "missing",
    }

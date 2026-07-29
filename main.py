"""
Main FastAPI entrypoint. Wires together the image and video routers.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Required environment variables:
    OPENAI_API_KEY                 - your OpenAI key (DO NOT hardcode it)
    GOOGLE_CLOUD_PROJECT            - GCP project id used for Veo/Gemini (Vertex AI)
    GOOGLE_CLOUD_REGION             - defaults to "us-central1" if unset
    (optional) GOOGLE_APPLICATION_CREDENTIALS - service account JSON key;
        for local dev, prefer: gcloud auth application-default login

Optional tuning:
    MEDIA_MAX_WORKERS               - concurrent image/video jobs (default 3)
    MEDIA_JOB_TTL_SECONDS           - prune completed/failed jobs after N seconds (default 3600)

Also requires the `ffmpeg` binary to be installed on the host machine.
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
from video_generation import router as video_router

app = FastAPI(title="Media Generation API", version="1.0.0")

# Allow your Flutter app (web/mobile/desktop) to call this API during dev.
# Lock this down to your actual domains before shipping to production.
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


@app.on_event("startup")
async def on_startup():
    # Warm the shared executor so first job doesn't pay setup cost on request path.
    from job_runtime import get_executor

    get_executor()
    logger.info(
        "Media Generation API ready (MEDIA_MAX_WORKERS=%s)",
        os.environ.get("MEDIA_MAX_WORKERS", "3"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

"""
Main FastAPI entrypoint. Wires together the image and video routers.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Required environment variables:
    OPENAI_API_KEY                 - your OpenAI key (DO NOT hardcode it)
    GOOGLE_CLOUD_PROJECT            - GCP project id used for Veo/Gemini (Vertex AI)
    GOOGLE_CLOUD_REGION             - defaults to "us-central1" if unset
    GOOGLE_APPLICATION_CREDENTIALS  - path to a service account JSON key

Also requires the `ffmpeg` binary to be installed on the host machine.
See README.md for full endpoint documentation.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

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


@app.get("/health")
async def health():
    return {"status": "ok"}

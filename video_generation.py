"""
Video generation module — Veo 3.1 + Gemini (Google Vertex AI).

Exposes:
  - router    FastAPI router with:
                POST /api/video/generate
                GET  /api/video/status/{job_id}

Required environment variables:
  GOOGLE_CLOUD_PROJECT
  GOOGLE_CLOUD_REGION (optional, defaults to "us-central1")
  GOOGLE_APPLICATION_CREDENTIALS (path to a service account JSON key)

Also requires the `ffmpeg` binary on the host.
"""

import os
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

SEGMENT_LEN = 8  # Veo 3.1 only accepts 4, 6, or 8 seconds per single call
_LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
_ALLOWED_DURATIONS = {8, 16, 30}

router = APIRouter(prefix="/api/video", tags=["video"])

# In-memory job store. Swap for Redis/DB if you run multiple server processes.
_jobs = {}
_jobs_lock = threading.Lock()


def _public_url(filename: str) -> str:
    return f"/files/{filename}"


def _set_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _get_genai_clients():
    from google import genai

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is not set.")
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

    client = genai.Client(enterprise=True, project=project_id, location=location)
    gemini_client = genai.Client(enterprise=True, project=project_id, location="global")
    return client, gemini_client


def extract_last_frame(video_path, frame_path):
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-1", "-i", video_path, "-update", "1", "-q:v", "1", frame_path],
        check=True,
        capture_output=True,
    )


def concat_and_trim(clip_paths, out_path, target_seconds):
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    concat_path = out_path + "_full.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", concat_path],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", concat_path, "-t", str(target_seconds), "-c", "copy", out_path],
        check=True,
        capture_output=True,
    )


def build_prompt(language_name, ad_text, segment_index, covered_context, camera_motion):
    if segment_index == 0:
        return f"""
You are an expert prompt engineer for Google's Veo video model, and a scriptwriter
for short recruitment/hiring advertisement videos.

Analyze the attached starting image (a job-recruitment poster/scene) and turn the
raw job-advertisement text below into ONE cohesive 8-second cinematic Veo prompt.

Requirements:
- The spoken voiceover/dialogue in the video must be written in {language_name},
  and should summarize the key hook of the ad (company name, that hiring is open,
  and urgency) in a natural, energetic recruiter/announcer voice.
- Include the camera motion keyword: {camera_motion}.
- Describe visual style, setting, motion, and mood, integrating the image's subject.
- Output ONLY the final Veo prompt text (including the {language_name} spoken line
  in quotes), no preamble, no markdown.

Raw ad text:
\"\"\"{ad_text}\"\"\"
"""
    return f"""
You are continuing an 8-second Veo video segment (segment #{segment_index + 1}) of a
recruitment advertisement. The previous segment ended on the attached frame.

Write the next 8-second Veo prompt that continues smoothly from that frame.

Requirements:
- Continue the voiceover in {language_name}, covering the NEXT chunk of the job ad
  content below that hasn't been spoken yet (e.g. next open positions, salary, or
  the location/contact number), staying energetic and clear.
- Keep visual style/character/setting consistent; camera motion may vary for
  cinematic variety.
- Output ONLY the final Veo prompt text (including the {language_name} spoken line
  in quotes), no preamble, no markdown.

Full ad text for reference (avoid repeating lines already used):
\"\"\"{ad_text}\"\"\"

Content already covered so far:
{covered_context}
"""


def generate_segment(client, gemini_client, gemini_model, video_model, image_path, prompt_text):
    from google.genai import types

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    gem_response = gemini_client.models.generate_content(
        model=gemini_model,
        contents=[prompt_text, types.Part.from_bytes(data=image_bytes, mime_type=mime)],
    )
    veo_prompt = gem_response.text.strip()

    operation = client.models.generate_videos(
        model=video_model,
        prompt=veo_prompt,
        image=types.Image.from_file(location=image_path),
        config=types.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            duration_seconds=SEGMENT_LEN,
            resolution="1080p",
            person_generation="allow_adult",
            generate_audio=True,
        ),
    )

    while not operation.done:
        time.sleep(15)
        operation = client.operations.get(operation)

    if not operation.response:
        raise RuntimeError("Video generation failed / returned no response.")

    return operation.result.generated_videos[0].video.video_bytes, veo_prompt


def _run_video_job(
    job_id: str,
    starting_image_path: str,
    ad_text: str,
    language_name: str,
    target_seconds: int,
    camera_motion: str,
):
    gemini_model = "gemini-3.5-flash"
    video_model = "veo-3.1-generate-001"

    try:
        _set_job(job_id, status="processing", progress="Initializing")
        client, gemini_client = _get_genai_clients()

        n_segments = max(1, -(-target_seconds // SEGMENT_LEN))  # ceil division

        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_paths = []
            covered_context = ""
            current_image = starting_image_path

            for i in range(n_segments):
                _set_job(job_id, progress=f"Generating segment {i + 1}/{n_segments}")
                prompt_text = build_prompt(language_name, ad_text, i, covered_context, camera_motion)
                video_bytes, used_prompt = generate_segment(
                    client, gemini_client, gemini_model, video_model, current_image, prompt_text
                )

                clip_path = os.path.join(tmp_dir, f"seg{i}.mp4")
                with open(clip_path, "wb") as f:
                    f.write(video_bytes)
                clip_paths.append(clip_path)
                covered_context += " " + used_prompt

                if i < n_segments - 1:
                    frame_path = os.path.join(tmp_dir, f"seg{i}_last.jpg")
                    extract_last_frame(clip_path, frame_path)
                    current_image = frame_path

            _set_job(job_id, progress="Finalizing video")
            filename = f"{job_id}.mp4"
            out_path = os.path.join(GENERATED_DIR, filename)

            if len(clip_paths) == 1:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_paths[0], "-t", str(target_seconds), "-c", "copy", out_path],
                    check=True,
                    capture_output=True,
                )
            else:
                concat_and_trim(clip_paths, out_path, target_seconds)

        _set_job(
            job_id,
            status="completed",
            progress="Done",
            video_url=_public_url(filename),
        )
    except Exception as exc:
        _set_job(job_id, status="failed", error=str(exc))
    finally:
        if os.path.exists(starting_image_path):
            os.remove(starting_image_path)


class VideoGenerateResponse(BaseModel):
    job_id: str
    status: str


class VideoStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    progress: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None


@router.post("/generate", response_model=VideoGenerateResponse)
async def api_generate_video(
    background_tasks: BackgroundTasks,
    starting_image: UploadFile = File(...),
    ad_text: str = Form(...),
    language: str = Form("Marathi"),
    duration_seconds: int = Form(30),
    camera_motion: str = Form("Zoom (In)"),
):
    """
    Starts an async video generation job (this can take several minutes).

    multipart/form-data fields:
      - starting_image (required): the first-frame image file
      - ad_text (required): raw ad copy to turn into a voiceover script
      - language: one of English, Hindi, Marathi (default Marathi)
      - duration_seconds: one of 8, 16, 30 (default 30)
      - camera_motion: e.g. "Zoom (In)", "Pan (left)", "Static Shot (or fixed)", etc.

    Returns a job_id. Poll GET /api/video/status/{job_id} until status is
    "completed" (or "failed"), then download video_url.
    """
    if language not in _LANGUAGE_CODES:
        raise HTTPException(400, f"language must be one of {list(_LANGUAGE_CODES)}")
    if duration_seconds not in _ALLOWED_DURATIONS:
        raise HTTPException(400, f"duration_seconds must be one of {sorted(_ALLOWED_DURATIONS)}")

    job_id = uuid.uuid4().hex
    suffix = os.path.splitext(starting_image.filename or "")[1] or ".jpg"
    tmp_image_path = os.path.join(tempfile.gettempdir(), f"start_{job_id}{suffix}")
    with open(tmp_image_path, "wb") as f:
        f.write(await starting_image.read())

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": None,
            "video_url": None,
            "error": None,
        }

    background_tasks.add_task(
        _run_video_job,
        job_id,
        tmp_image_path,
        ad_text,
        language,
        duration_seconds,
        camera_motion,
    )

    return VideoGenerateResponse(job_id=job_id, status="queued")


@router.get("/status/{job_id}", response_model=VideoStatusResponse)
async def api_video_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job_id not found")
    return VideoStatusResponse(**job)

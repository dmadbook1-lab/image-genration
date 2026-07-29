"""
Video generation module — Veo 3.1 + Gemini (Google Vertex AI).

Exposes:
  - router    FastAPI router with:
                POST /api/video/generate
                GET  /api/video/status/{job_id}

Required environment variables:
  GOOGLE_CLOUD_PROJECT
  GOOGLE_CLOUD_REGION (optional, defaults to "us-central1")

Authenticate locally with Application Default Credentials (recommended):
  gcloud auth login
  gcloud config set project <your-project-id>
  gcloud auth application-default login

Also requires the `ffmpeg` binary on the host.
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from job_runtime import create_job, get_job, submit_job, update_job
from s3_storage import normalize_user_id, store_generated_media

GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

SEGMENT_LEN = 8  # Veo 3.1 only accepts 4, 6, or 8 seconds per single call
_LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
_ALLOWED_DURATIONS = {8, 16, 30}

router = APIRouter(prefix="/api/video", tags=["video"])

_genai_clients = None
_genai_clients_key: Optional[tuple[str, str]] = None
_genai_clients_lock = threading.Lock()


def _get_genai_clients():
    """Return cached Vertex GenAI clients; rebuild if project/region change."""
    global _genai_clients, _genai_clients_key
    from google import genai

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is not set.")
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    key = (project_id, location)

    if _genai_clients is not None and _genai_clients_key == key:
        return _genai_clients

    with _genai_clients_lock:
        if _genai_clients is not None and _genai_clients_key == key:
            return _genai_clients
        client = genai.Client(vertexai=True, project=project_id, location=location)
        gemini_client = genai.Client(vertexai=True, project=project_id, location="global")
        _genai_clients = (client, gemini_client)
        _genai_clients_key = key
        return _genai_clients


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


def _wait_for_video_operation(client, operation, poll_seconds=15, max_wait=900):
    elapsed = 0
    while not operation.done:
        if elapsed >= max_wait:
            raise RuntimeError(f"Video generation timed out after {max_wait}s")
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        operation = client.operations.get(operation)
    return operation


def _video_operation_payload(operation):
    if operation.error:
        raise RuntimeError(f"Video generation failed: {operation.error}")

    payload = operation.response or operation.result
    if payload is None:
        raise RuntimeError("Video generation completed but returned no response.")

    generated_videos = payload.generated_videos
    if not generated_videos:
        reasons = payload.rai_media_filtered_reasons or []
        count = payload.rai_media_filtered_count
        detail = ""
        if reasons:
            detail = f" RAI reasons: {reasons}"
        elif count is not None:
            detail = f" RAI filtered count: {count}"
        raise RuntimeError(f"Video generation returned no videos.{detail}")

    return generated_videos[0]


def _download_video_bytes(video) -> bytes:
    if video.video_bytes:
        return video.video_bytes

    uri = video.uri
    if not uri:
        raise RuntimeError("Generated video has no bytes or URI.")

    if uri.startswith("gs://"):
        from google.cloud import storage

        _, _, rest = uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise RuntimeError(f"Invalid GCS URI: {uri}")
        return storage.Client().bucket(bucket_name).blob(blob_name).download_as_bytes()

    import httpx

    response = httpx.get(uri, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    return response.content


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
    veo_prompt = (gem_response.text or "").strip()
    if not veo_prompt:
        raise RuntimeError("Gemini returned an empty Veo prompt.")

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

    operation = _wait_for_video_operation(client, operation)
    generated = _video_operation_payload(operation)
    if generated.video is None:
        raise RuntimeError("Generated video entry is missing video data.")

    video_bytes = _download_video_bytes(generated.video)
    return video_bytes, veo_prompt


def _run_video_job(
    job_id: str,
    starting_image_path: str,
    ad_text: str,
    language_name: str,
    target_seconds: int,
    camera_motion: str,
    user_id: str,
):
    gemini_model = "gemini-2.5-flash"
    video_model = "veo-3.1-lite-generate-001"

    try:
        update_job(job_id, status="processing", progress="Initializing", user_id=user_id)
        client, gemini_client = _get_genai_clients()

        n_segments = max(1, -(-target_seconds // SEGMENT_LEN))  # ceil division

        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_paths = []
            covered_context = ""
            current_image = starting_image_path

            for i in range(n_segments):
                update_job(job_id, progress=f"Generating segment {i + 1}/{n_segments}")
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

            update_job(job_id, progress="Finalizing video")
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

        update_job(job_id, progress="Uploading to S3")
        video_url, s3_key = store_generated_media(
            local_path=out_path,
            user_id=user_id,
            media_kind="videos",
            filename=filename,
            content_type="video/mp4",
            delete_local=True,
        )

        update_job(
            job_id,
            status="completed",
            progress="Done",
            video_url=video_url,
            s3_key=s3_key,
            user_id=user_id,
        )
    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        update_job(job_id, status="failed", error=str(exc))
    finally:
        if os.path.exists(starting_image_path):
            os.remove(starting_image_path)


class VideoGenerateResponse(BaseModel):
    job_id: str
    status: str
    user_id: str


class VideoStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    progress: Optional[str] = None
    video_url: Optional[str] = None
    s3_key: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/generate", response_model=VideoGenerateResponse)
async def api_generate_video(
    starting_image: UploadFile = File(...),
    ad_text: str = Form(...),
    user_id: str = Form(...),
    language: str = Form("Marathi"),
    duration_seconds: int = Form(30),
    camera_motion: str = Form("Zoom (In)"),
):
    """
    Starts an async video generation job on a worker thread (can take several minutes).
    On completion, stores under users/{user_id}/generated/videos/ on S3.

    multipart/form-data fields:
      - starting_image (required): the first-frame image file
      - ad_text (required): raw ad copy to turn into a voiceover script
      - user_id (required): logged-in AdvPost user id
      - language: one of English, Hindi, Marathi (default Marathi)
      - duration_seconds: one of 8, 16, 30 (default 30)
      - camera_motion: e.g. "Zoom (In)", "Pan (left)", "Static Shot (or fixed)", etc.

    Returns a job_id. Poll GET /api/video/status/{job_id} until status is
    "completed" (or "failed"), then use video_url (S3 public URL).
    """
    if language not in _LANGUAGE_CODES:
        raise HTTPException(400, f"language must be one of {list(_LANGUAGE_CODES)}")
    if duration_seconds not in _ALLOWED_DURATIONS:
        raise HTTPException(400, f"duration_seconds must be one of {sorted(_ALLOWED_DURATIONS)}")

    try:
        uid = normalize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    suffix = os.path.splitext(starting_image.filename or "")[1] or ".jpg"
    tmp_image_path = os.path.join(tempfile.gettempdir(), f"start_{job_id}{suffix}")
    with open(tmp_image_path, "wb") as f:
        f.write(await starting_image.read())

    create_job(
        job_id,
        status="queued",
        progress=None,
        video_url=None,
        s3_key=None,
        user_id=uid,
        error=None,
    )

    submit_job(
        _run_video_job,
        job_id,
        tmp_image_path,
        ad_text,
        language,
        duration_seconds,
        camera_motion,
        uid,
    )

    return VideoGenerateResponse(job_id=job_id, status="queued", user_id=uid)


@router.get("/status/{job_id}", response_model=VideoStatusResponse)
async def api_video_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "job_id not found")
    return VideoStatusResponse(
        job_id=job["job_id"],
        status=job.get("status", "queued"),
        progress=job.get("progress"),
        video_url=job.get("video_url"),
        s3_key=job.get("s3_key"),
        user_id=job.get("user_id"),
        error=job.get("error"),
    )

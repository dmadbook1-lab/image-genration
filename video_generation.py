"""
Video generation module — Veo 3.1 + Gemini (Google Vertex AI).

Exposes:
  - router    FastAPI router with:
                POST /api/video/generate
                GET  /api/video/status/{job_id}

Auth: service account JSON (default file beside this module). Override with
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json

Optional environment variables:
  GOOGLE_CLOUD_PROJECT  (defaults to project_id inside the SA JSON)
  GOOGLE_CLOUD_REGION   (defaults to "us-central1")

Also requires the `ffmpeg` binary on the host.
"""

import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from job_runtime import create_job, get_job, submit_job, update_job
from s3_storage import normalize_user_id, store_generated_media

_MODULE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = os.path.join(_MODULE_DIR, "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# Default Veo service account key shipped with this service.
_DEFAULT_SA_JSON = _MODULE_DIR / "video-generation-veo-502109-7f1a9e95d0c7.json"

logger = logging.getLogger(__name__)

SEGMENT_LEN = 8  # Veo 3.1 only accepts 4, 6, or 8 seconds per single call
_LANGUAGE_CODES = {"English": "en", "Hindi": "hi", "Marathi": "mr"}
_ALLOWED_DURATIONS = {8, 16, 30}

router = APIRouter(prefix="/api/video", tags=["video"])

_genai_clients = None
_genai_clients_key: Optional[tuple[str, str, str]] = None
_genai_clients_lock = threading.Lock()
_vertex_credentials = None
_vertex_credentials_path: Optional[str] = None
_vertex_credentials_lock = threading.Lock()


def _service_account_json_path() -> Path:
    override = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = _MODULE_DIR / path
        return path.resolve()
    return _DEFAULT_SA_JSON


def _get_vertex_credentials():
    """Load Vertex credentials from the Veo service-account JSON."""
    global _vertex_credentials, _vertex_credentials_path
    from google.oauth2 import service_account

    path = _service_account_json_path()
    path_str = str(path)
    if _vertex_credentials is not None and _vertex_credentials_path == path_str:
        return _vertex_credentials

    with _vertex_credentials_lock:
        if _vertex_credentials is not None and _vertex_credentials_path == path_str:
            return _vertex_credentials
        if not path.is_file():
            raise RuntimeError(
                f"GCP service account JSON not found at {path}. "
                "Place video-generation-veo-502109-7f1a9e95d0c7.json beside "
                "video_generation.py or set GOOGLE_APPLICATION_CREDENTIALS."
            )
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        creds = service_account.Credentials.from_service_account_file(
            path_str, scopes=scopes
        )
        _vertex_credentials = creds
        _vertex_credentials_path = path_str
        return _vertex_credentials


def _get_vertex_project_id(credentials) -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if project_id:
        return project_id
    project_id = getattr(credentials, "project_id", None)
    if project_id:
        return project_id
    raise RuntimeError(
        "GOOGLE_CLOUD_PROJECT is not set and the service account JSON "
        "has no project_id."
    )


def _get_genai_clients():
    """Return cached Vertex GenAI clients authenticated via the SA JSON."""
    global _genai_clients, _genai_clients_key
    from google import genai

    credentials = _get_vertex_credentials()
    project_id = _get_vertex_project_id(credentials)
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    key = (project_id, location, _vertex_credentials_path or "")

    if _genai_clients is not None and _genai_clients_key == key:
        return _genai_clients

    with _genai_clients_lock:
        if _genai_clients is not None and _genai_clients_key == key:
            return _genai_clients
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            credentials=credentials,
        )
        gemini_client = genai.Client(
            vertexai=True,
            project=project_id,
            location="global",
            credentials=credentials,
        )
        _genai_clients = (client, gemini_client)
        _genai_clients_key = key
        return _genai_clients


def ffmpeg_available() -> bool:
    """True if the ffmpeg binary is on PATH."""
    from shutil import which

    return which("ffmpeg") is not None


def extract_last_frame(video_path, frame_path):
    _run_ffmpeg(
        [
            "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
            "-update", "1", "-q:v", "1", frame_path,
        ],
        label="extract last frame",
    )


def _run_ffmpeg(cmd: list[str], *, label: str) -> None:
    """Run ffmpeg and raise a readable error with stderr on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed on this server (or not on PATH). "
            "Install it and restart the service — "
            "Ubuntu/Debian: sudo apt-get install -y ffmpeg | "
            "RHEL/Amazon Linux: sudo yum install -y ffmpeg | "
            "macOS: brew install ffmpeg"
        ) from exc

    if result.returncode == 0:
        return

    stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
    # FFmpeg often maps ENOSPC (28) to exit 228 (256 - 28).
    disk_hint = ""
    if result.returncode in (228, 28) or "No space left" in stderr:
        disk_hint = " Disk appears full — free space on the server and retry."

    tail = stderr[-800:] if stderr else "(no ffmpeg stderr)"
    raise RuntimeError(
        f"ffmpeg {label} failed (exit {result.returncode}).{disk_hint}\n{tail}"
    )


def concat_and_trim(clip_paths, out_path, target_seconds, work_dir=None):
    """Concatenate segment clips, then trim to target_seconds.

    Prefers stream-copy for speed; falls back to re-encode when codecs differ.
    Intermediate files stay in work_dir (or next to out_path).
    """
    work = work_dir or os.path.dirname(out_path) or "."
    os.makedirs(work, exist_ok=True)

    list_file = os.path.join(work, "concat_list.txt")
    concat_path = os.path.join(work, "concat_full.mp4")

    with open(list_file, "w") as f:
        for p in clip_paths:
            # Escape single quotes for the concat demuxer.
            abs_path = os.path.abspath(p).replace("'", r"'\''")
            f.write(f"file '{abs_path}'\n")

    copy_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", concat_path,
    ]
    try:
        _run_ffmpeg(copy_cmd, label="concat (copy)")
    except RuntimeError as copy_err:
        # Veo segments can differ in codec/params — re-encode as a fallback.
        logger.warning("Stream-copy concat failed, re-encoding: %s", copy_err)
        reencode_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            concat_path,
        ]
        _run_ffmpeg(reencode_cmd, label="concat (re-encode)")

    trim_cmd = [
        "ffmpeg", "-y", "-i", concat_path,
        "-t", str(target_seconds),
        "-c", "copy",
        out_path,
    ]
    try:
        _run_ffmpeg(trim_cmd, label="trim (copy)")
    except RuntimeError:
        trim_reencode = [
            "ffmpeg", "-y", "-i", concat_path,
            "-t", str(target_seconds),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            out_path,
        ]
        _run_ffmpeg(trim_reencode, label="trim (re-encode)")


def _wait_for_video_operation(client, operation, poll_seconds=8, max_wait=900):
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
        credentials = _get_vertex_credentials()
        gcs = storage.Client(
            project=_get_vertex_project_id(credentials),
            credentials=credentials,
        )
        return gcs.bucket(bucket_name).blob(blob_name).download_as_bytes()

    import httpx

    response = httpx.get(uri, follow_redirects=True, timeout=120.0)
    response.raise_for_status()
    return response.content


_SAFETY_RULES = """
CRITICAL SAFETY / RAI RULES (must follow — Veo blocks photorealistic people):
- Do NOT depict any human faces, celebrities, public figures, children, or
  photorealistic people of any kind.
- Prefer: products, packaging, storefronts, interiors, food, logos as graphics,
  text overlays, hands-only close-ups (no face), city/street atmosphere without
  identifiable faces, abstract motion graphics.
- Never name or describe a real person. Never invent celebrity look-alikes.
- Keep the ad family-friendly: no violence, weapons, sexual content, alcohol
  abuse, drugs, hate, or political content.
- Voiceover may be an off-screen announcer — do not show the speaker on camera.
""".strip()

_MOTION_RULES_TEMPLATE = """
MOTION REQUIREMENTS (critical — the previous videos were coming out too static):
- The video must show continuous, visible motion for the FULL 8 seconds, not one
  frozen composition with a voiceover played over it.
- Camera motion for this segment: {camera_motion}. If this is "Static Shot" or
  "fixed", treat it as meaning the camera rig doesn't move — it does NOT mean the
  scene is frozen. You must still animate the scene itself.
- In addition to (or instead of) camera movement, describe specific in-scene
  motion appropriate to the brief: product rotating or being picked up/opened,
  liquid pouring, steam rising, packaging unboxing, fabric or hair-free motion
  graphics, light/reflections shifting, particles or dust in the air, screens or
  UI elements animating, text overlays sliding/fading in, vehicles or machinery
  moving, storefront signage lighting up, etc. Pick 1-2 that fit the business.
- Include at least one visible change over the 8 seconds — a shift in framing,
  distance, subject focus, or a clear beat/transition (e.g. "opens on a wide
  shot of the storefront, then pushes in as the product is revealed on the
  counter") — never one static, held, or frozen composition throughout.
- Do not use the words "static," "frozen," "still," or "held" to describe the
  overall shot — only ever to describe camera rig stability, if at all.
""".strip()


def build_prompt(
    language_name,
    ad_text,
    segment_index,
    covered_context,
    camera_motion,
    starting_image_type="Scene",
    has_starting_image=False,
    is_final_segment=True,
    safe_mode=False,
):
    if has_starting_image:
        type_label = (starting_image_type or "Scene").strip() or "Scene"
        assets_block = (
            f"- The attached image is a {type_label.lower()} — treat it as the "
            "visual anchor for the first frame / continuity. If it contains a "
            "person or face, reinterpret it as a logo/product graphic only and "
            "do not animate a photorealistic person."
        )
    else:
        assets_block = (
            "- No reference image was provided — invent a cinematic promotional "
            "scene with products, storefront, packaging, or motion graphics "
            "(NO people / NO faces)."
        )

    brief_rules = """
The advertisement brief below may be a structured brief with labeled sections
(BUSINESS DETAILS, AD MESSAGE, VISUAL STYLE, CONTACT DETAILS) or plain ad text.
Honor every section that is present:
- BUSINESS DETAILS: mention the business name in the voiceover; let the category,
  purpose, and target audience shape the scene, tone, and wording.
- AD MESSAGE: this is the core of the voiceover script. If it asks you to write
  the script yourself, write a compelling one from the business details.
- VISUAL STYLE: follow this style/mood with products and environments only
  (never add people to make a style feel "friendly").
- CONTACT DETAILS: work them into the closing call-to-action (spoken naturally)
  and describe clean on-screen text overlays showing them near the end of the
  video. Reproduce phone numbers, websites, and addresses EXACTLY as written —
  never invent or alter contact information.""".strip()

    safe_extra = ""
    if safe_mode:
        safe_extra = (
            "\nSAFE RETRY MODE: Previous attempt was blocked by Vertex AI safety "
            "filters. Rewrite as a purely product / storefront / text-overlay "
            "commercial with ZERO humans, faces, or celebrity likenesses.\n"
        )

    # FIX: this must be computed unconditionally (not inside `if is_final_segment`)
    # since it's referenced by every return branch below, for every segment.
    motion_rules = _MOTION_RULES_TEMPLATE.format(camera_motion=camera_motion)

    final_rule = ""
    if is_final_segment:
        final_rule = (
            "- This is the FINAL segment: end with a strong call-to-action, and "
            "if the brief has CONTACT DETAILS, speak them naturally and show "
            "them as clean on-screen text.\n"
        )

    if segment_index == 0:
        return f"""
You are an expert prompt engineer for Google's Veo video model, and a scriptwriter
for short promotional / recruitment advertisement videos.

Turn the advertisement brief below into ONE cohesive 8-second cinematic Veo prompt.

{_SAFETY_RULES}
{safe_extra}
{brief_rules}

Image asset guidance:
{assets_block}

Requirements:
- The spoken voiceover/dialogue in the video must be written in {language_name},
  and should summarize the key hook of the ad (company/brand name, offer or hiring
  message, and urgency) in a natural, energetic announcer voice (off-screen).
{motion_rules}
- Describe visual style, setting, and mood clearly — products and places only.
{final_rule}- Output ONLY the final Veo prompt text (including the {language_name} spoken line
  in quotes), no preamble, no markdown.

Advertisement brief:
\"\"\"{ad_text}\"\"\"
"""
    return f"""
You are continuing an 8-second Veo video segment (segment #{segment_index + 1}) of a
promotional / recruitment advertisement. The previous segment ended on the attached frame.

Write the next 8-second Veo prompt that continues smoothly from that frame.

{_SAFETY_RULES}
{safe_extra}
{brief_rules}

Image asset guidance (keep continuity):
{assets_block}

Requirements:
- Continue the voiceover in {language_name}, covering the NEXT chunk of the ad
  content below that hasn't been spoken yet, staying energetic and clear.
{final_rule}{motion_rules}
- Keep visual style/setting consistent across segments. Still NO people or faces.
- Output ONLY the final Veo prompt text (including the {language_name} spoken line
  in quotes), no preamble, no markdown.

Full advertisement brief for reference (avoid repeating lines already used):
\"\"\"{ad_text}\"\"\"

Content already covered so far:
{covered_context}
"""

def _image_part(path: str):
    from google.genai import types

    with open(path, "rb") as f:
        image_bytes = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return types.Part.from_bytes(data=image_bytes, mime_type=mime)


def _is_rai_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rai" in msg or "usage guidelines" in msg or "filtered out" in msg


def _gemini_veo_prompt(gemini_client, gemini_model, prompt_text, image_path=None) -> str:
    contents = [prompt_text]
    if image_path and os.path.exists(image_path):
        contents.append(_image_part(image_path))

    gem_response = gemini_client.models.generate_content(
        model=gemini_model,
        contents=contents,
    )
    veo_prompt = (gem_response.text or "").strip()
    if not veo_prompt:
        raise RuntimeError("Gemini returned an empty Veo prompt.")
    return veo_prompt


def _call_veo(client, video_model, veo_prompt, image_path=None):
    from google.genai import types

    # 9:16 + 720p: reel feed crop, faster Veo, smaller download/upload.
    video_kwargs = {
        "model": video_model,
        "prompt": veo_prompt,
        "config": types.GenerateVideosConfig(
            aspect_ratio="9:16",
            number_of_videos=1,
            duration_seconds=SEGMENT_LEN,
            resolution="720p",
            # Photorealistic people trigger celebrity/person RAI filters
            # (support code 15236754) on many Vertex projects.
            person_generation="dont_allow",
            generate_audio=True,
        ),
    }
    if image_path and os.path.exists(image_path):
        video_kwargs["image"] = types.Image.from_file(location=image_path)

    operation = client.models.generate_videos(**video_kwargs)
    operation = _wait_for_video_operation(client, operation)
    generated = _video_operation_payload(operation)
    if generated.video is None:
        raise RuntimeError("Generated video entry is missing video data.")
    return _download_video_bytes(generated.video)


def generate_segment(
    client,
    gemini_client,
    gemini_model,
    video_model,
    image_path,
    prompt_text,
    *,
    safe_retry_prompt_builder=None,
):
    """Generate one Veo segment. On RAI filter, auto-retry with a safer prompt
    (and without the starting image if needed).
    """
    veo_prompt = _gemini_veo_prompt(
        gemini_client, gemini_model, prompt_text, image_path=image_path
    )
    logger.info("Veo prompt (first attempt): %s", veo_prompt[:500])

    try:
        video_bytes = _call_veo(client, video_model, veo_prompt, image_path=image_path)
        return video_bytes, veo_prompt
    except Exception as first_err:
        if not _is_rai_error(first_err):
            raise
        logger.warning("Veo RAI filter hit; retrying with safer no-people prompt: %s", first_err)

    # Retry 1: safer Gemini rewrite, keep image
    if safe_retry_prompt_builder is not None:
        safe_prompt_text = safe_retry_prompt_builder()
    else:
        safe_prompt_text = (
            prompt_text
            + "\n\nSAFE RETRY: rewrite with ZERO humans/faces; products and "
            "storefronts only."
        )
    veo_prompt = _gemini_veo_prompt(
        gemini_client, gemini_model, safe_prompt_text, image_path=None
    )
    logger.info("Veo prompt (safe retry): %s", veo_prompt[:500])

    try:
        video_bytes = _call_veo(client, video_model, veo_prompt, image_path=None)
        return video_bytes, veo_prompt
    except Exception as second_err:
        if not _is_rai_error(second_err):
            raise
        logger.warning("Veo RAI filter hit again on safe retry: %s", second_err)
        raise RuntimeError(
            "Video was blocked by Vertex AI safety filters (often for "
            "photorealistic people / celebrity likeness). Try again with a "
            "product or storefront image, avoid describing people, and keep "
            "the ad message family-friendly."
        ) from second_err



def _run_video_job(
    job_id: str,
    starting_image_path: Optional[str],
    ad_text: str,
    language_name: str,
    target_seconds: int,
    camera_motion: str,
    user_id: str,
    starting_image_type: str = "Scene",
):
    gemini_model = "gemini-2.5-flash"
    video_model = "veo-3.1-lite-generate-001"

    try:
        update_job(job_id, status="processing", progress="Initializing", user_id=user_id)
        client, gemini_client = _get_genai_clients()

        n_segments = max(1, -(-target_seconds // SEGMENT_LEN))  # ceil division
        has_starting = bool(starting_image_path and os.path.exists(starting_image_path))

        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_paths = []
            covered_context = ""
            current_image = starting_image_path if has_starting else None

            for i in range(n_segments):
                update_job(job_id, progress=f"Generating segment {i + 1}/{n_segments}")
                prompt_kwargs = dict(
                    language_name=language_name,
                    ad_text=ad_text,
                    segment_index=i,
                    covered_context=covered_context,
                    camera_motion=camera_motion,
                    starting_image_type=starting_image_type,
                    has_starting_image=bool(current_image and os.path.exists(current_image)),
                    is_final_segment=(i == n_segments - 1),
                )
                prompt_text = build_prompt(**prompt_kwargs)

                def _safe_builder(
                    _kwargs=prompt_kwargs,
                ):
                    return build_prompt(**{**_kwargs, "safe_mode": True, "has_starting_image": False})

                video_bytes, used_prompt = generate_segment(
                    client,
                    gemini_client,
                    gemini_model,
                    video_model,
                    current_image,
                    prompt_text,
                    safe_retry_prompt_builder=_safe_builder,
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
                _run_ffmpeg(
                    [
                        "ffmpeg", "-y", "-i", clip_paths[0],
                        "-t", str(target_seconds), "-c", "copy", out_path,
                    ],
                    label="single-clip trim",
                )
            else:
                concat_and_trim(
                    clip_paths,
                    out_path,
                    target_seconds,
                    work_dir=tmp_dir,
                )

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
            filename=filename,
            user_id=user_id,
        )
    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        update_job(job_id, status="failed", error=str(exc))
    finally:
        if starting_image_path and os.path.exists(starting_image_path):
            try:
                os.remove(starting_image_path)
            except OSError:
                pass


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
    filename: Optional[str] = None
    user_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/generate", response_model=VideoGenerateResponse)
async def api_generate_video(
    ad_text: str = Form(...),
    user_id: str = Form(...),
    language: str = Form("Marathi"),
    duration_seconds: int = Form(8),
    camera_motion: str = Form("Zoom (In)"),
    starting_image_type: str = Form("Scene"),
    starting_image: Optional[UploadFile] = File(None),
):
    """
    Starts an async video generation job on a worker thread (can take several minutes).
    On completion, stores under users/{user_id}/generated/videos/ on S3.

    multipart/form-data fields:
      - starting_image (optional): first-frame image (Scene/Logo/Product)
      - starting_image_type: Scene | Logo | Product (default Scene)
      - ad_text (required): raw ad copy to turn into a voiceover script
      - user_id (required): logged-in AdvPost user id
      - language: one of English, Hindi, Marathi (default Marathi)
      - duration_seconds: one of 8, 16, 30 (default 8)
      - camera_motion: e.g. "Zoom (In)", "Pan (left)", "Static Shot (or fixed)", etc.

    Returns a job_id. Poll GET /api/video/status/{job_id} until status is
    "completed" (or "failed"), then use video_url (S3 public URL).
    """
    if language not in _LANGUAGE_CODES:
        raise HTTPException(400, f"language must be one of {list(_LANGUAGE_CODES)}")
    if duration_seconds not in _ALLOWED_DURATIONS:
        raise HTTPException(400, f"duration_seconds must be one of {sorted(_ALLOWED_DURATIONS)}")

    # Character encourages people shots that Veo RAI blocks (person_generation=dont_allow).
    allowed_types = {"Scene", "Logo", "Product", "Character"}
    image_type = (starting_image_type or "Scene").strip()
    if image_type not in allowed_types:
        raise HTTPException(400, f"starting_image_type must be one of {sorted(allowed_types)}")
    if image_type == "Character":
        image_type = "Product"

    try:
        uid = normalize_user_id(user_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    tmp_image_path = None
    if starting_image is not None and starting_image.filename:
        suffix = os.path.splitext(starting_image.filename or "")[1] or ".jpg"
        tmp_image_path = os.path.join(tempfile.gettempdir(), f"start_{job_id}{suffix}")
        data = await starting_image.read()
        if data:
            with open(tmp_image_path, "wb") as f:
                f.write(data)
        else:
            tmp_image_path = None

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
        image_type,
    )

    return VideoGenerateResponse(job_id=job_id, status="queued", user_id=uid)


@router.get("/status/{job_id}", response_model=VideoStatusResponse)
async def api_video_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, "job_id not found")
    filename = job.get("filename")
    if not filename and job.get("video_url"):
        # Fallback so local /files rewrite works even for older jobs.
        filename = f"{job_id}.mp4"

    return VideoStatusResponse(
        job_id=job["job_id"],
        status=job.get("status", "queued"),
        progress=job.get("progress"),
        video_url=job.get("video_url"),
        s3_key=job.get("s3_key"),
        filename=filename,
        user_id=job.get("user_id"),
        error=job.get("error"),
    )

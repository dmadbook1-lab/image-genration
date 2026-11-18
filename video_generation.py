"""
Video generation module — Veo 3.1 + Gemini (Google Vertex AI).

Plan-first architecture:
  1. Gemini writes a structured JSON ad plan (scenes, camera moves, voiceover,
     hashtags, on-screen text) based on business details.
  2. Gemini writes one voiceover line per scene, matched to that scene's
     visuals and sized to fit its duration, so each scene's Veo prompt asks
     Veo to speak its own line aloud (generate_audio=True).
  3. Each scene gets its own focused, category-aware Veo prompt with safety
     rules and per-category "suggested movement" library.
  4. Scenes are generated in parallel (fast_mode) or sequentially
     (frame-chained for pixel continuity).
  5. Clips are concatenated, trimmed to target length, and human-readable
     on-screen text (business name / offer / CTA / phone) is overlaid by
     ffmpeg drawtext — never baked into the AI video.

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

import concurrent.futures
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from generation_logger import dumps_json, log_generation_event
from job_runtime import create_job, get_job, submit_job, update_job
from s3_storage import normalize_user_id, store_generated_media

_MODULE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = os.path.join(_MODULE_DIR, "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# Default Veo service account key shipped with this service.
_DEFAULT_SA_JSON = _MODULE_DIR / "advpost-video-27ad9b6f9358.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VEO_ALLOWED_SEGMENT_SECONDS = (4, 6, 8)  # only these are accepted per Veo call
# Language name → ISO-ish code (used for allowlist validation; voiceover uses the name).
# Includes English + the 22 scheduled languages of India.
_LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Urdu": "ur",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Odia": "or",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Assamese": "as",
    "Maithili": "mai",
    "Santali": "sat",
    "Kashmiri": "ks",
    "Nepali": "ne",
    "Konkani": "kok",
    "Sindhi": "sd",
    "Dogri": "doi",
    "Manipuri": "mni",
    "Bodo": "brx",
    "Sanskrit": "sa",
}
# Durations exposed by the Flutter app (create_reel_flow_provider.apiDurationSeconds)
_ALLOWED_DURATIONS = {8, 16, 30}

VIDEO_MODEL = "veo-3.1-fast-generate-001"
GEMINI_MODEL = "gemini-3.5-flash"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10
# Gemini/Vertex quota errors (429 RESOURCE_EXHAUSTED) reset on a per-minute
# window, not instantly — the plain 10s/20s backoff below can burn through
# all retries before that window rolls over. Wait long enough to actually
# clear it instead of failing the whole job on a transient burst.
QUOTA_RETRY_BACKOFF_SECONDS = 35

# ffmpeg binary to run. Point FFMPEG_BINARY at a build with drawtext
# (libfreetype) when the one on PATH lacks it, e.g. Homebrew's keg-only
# /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg.
FFMPEG_BIN = os.environ.get("FFMPEG_BINARY", "").strip() or "ffmpeg"

router = APIRouter(prefix="/api/video", tags=["video"])

_genai_clients = None
_genai_clients_key: Optional[tuple[str, str, str]] = None
_genai_clients_lock = threading.Lock()
_vertex_credentials = None
_vertex_credentials_path: Optional[str] = None
_vertex_credentials_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Camera motion options
# ---------------------------------------------------------------------------
_CAMERA_MOTION_OPTIONS = [
    "None", "Eye-Level Shot", "Low-Angle Shot", "High-Angle Shot", "Zoom (In)",
    "Zoom (Out)", "Dolly (In)", "Dolly (Out)", "Pan (left)", "Pan (right)",
    "Crane Shot", "Aerial Shot", "Static Shot (or fixed)",
]

_LANGUAGE_CTA = {
    "English": "Call Now",
    "Hindi": "Abhi Call Karein",
    "Bengali": "Ekhon Call Korun",
    "Telugu": "Ippudu Call Cheyandi",
    "Marathi": "Aatach Call Kara",
    "Tamil": "Ippe Call Seyunga",
    "Urdu": "Abhi Call Karein",
    "Gujarati": "Have Call Karo",
    "Kannada": "Eega Call Maadi",
    "Odia": "Ebe Call Karantu",
    "Malayalam": "Ippol Call Cheyyu",
    "Punjabi": "Hun Call Karo",
    "Assamese": "Etiya Call Korok",
    "Maithili": "Abhi Call Karu",
    "Santali": "Call Now",
    "Kashmiri": "Call Now",
    "Nepali": "Ahile Call Garnuhos",
    "Konkani": "Atam Call Kara",
    "Sindhi": "Call Now",
    "Dogri": "Call Now",
    "Manipuri": "Call Now",
    "Bodo": "Call Now",
    "Sanskrit": "Call Now",
}

# ---------------------------------------------------------------------------
# Safety & motion rule templates
# ---------------------------------------------------------------------------
_SAFETY_RULES = """
CRITICAL SAFETY RULES (must follow):
- Do NOT depict any real, named, or identifiable public figure, celebrity, or
  politician, and do not invent a celebrity look-alike.
- People shown must be generic, non-identifiable models/actors appropriate to
  the scene (e.g. "a stylist", "a customer", "a farmer") — never a specific
  real person.
- Keep everything family-friendly: no violence, weapons, sexual content,
  alcohol/drug abuse, hate, or political content.
- No children shown without a supervising adult also in frame.
""".strip()

_MOTION_RULES_TEMPLATE = """
VIDEO DIRECTION (VERY IMPORTANT)

This video must feel like a professionally directed commercial.
Never create a static or frozen scene.
The scene must remain visually alive from the first frame to the last frame.

Camera Direction:
{camera_motion_desc}

Scene Direction:
- Every scene must contain continuous natural movement.
- The subject should actively perform meaningful actions related to the business.
- Background elements should also have subtle natural motion.
- The camera movement should feel cinematic and intentional.
- Every 2-3 seconds something new should happen.

Suggested movement for this business type:
{business_movement_examples}

Motion Rules:
- Never hold the exact same composition for more than 2 seconds.
- Add natural body movement, realistic facial expressions, environmental
  movement, and smooth camera transitions.
- Keep movement realistic and physically accurate; avoid repetitive or
  looping animations. Never freeze the background or the subject.

Audio rule:
- This scene must include a spoken voiceover line, delivered in a natural,
  energetic announcer voice, in addition to ambient/background sound. See
  the "Voiceover for this scene" section below for exactly what to say.

Do NOT generate a slideshow, a talking image, or a static product shot.
Every second should feel like a real advertisement shot by a professional
film crew.
""".strip()

# ---------------------------------------------------------------------------
# Per-category movement library
# ---------------------------------------------------------------------------
BUSINESS_MOTION_LIBRARY = {
    "salon": ["Customer entering salon", "Reception greeting", "Hair wash",
              "Hair cutting", "Hair styling", "Happy customer smiling"],
    "beauty wellness": ["Customer relaxing during treatment", "Therapist applying product",
                         "Product being poured/applied", "Client smiling in mirror"],
    "restaurant cafe": ["Chef cooking", "Food plating", "Steam rising", "Serving food",
                         "Family enjoying meal"],
    "food beverage": ["Ingredients being prepared", "Product being poured/packaged",
                       "Steam or condensation", "Customer tasting/enjoying product"],
    "agriculture": ["Farmer inspecting crops", "Spraying fertilizer", "Drone monitoring field",
                     "Healthy crop close-up", "Harvest scene"],
    "health fitness": ["Trainer demonstrating exercise", "Client working out",
                        "Equipment in motion", "Sweat/effort close-up", "High-five after workout"],
    "medical": ["Doctor consulting patient", "Medicine handover", "Reception assistance",
                "Pharmacist arranging medicines"],
    "stores services": ["Customer shopping", "Product pickup", "Cash counter billing",
                         "Shopping bags moving", "Staff assisting customer"],
    "products": ["Product being unboxed", "Close-up product rotation",
                 "Hands demonstrating use", "Packaging being sealed/labelled"],
    "construction real estate": ["Workers on site", "Crane or machinery moving",
                                  "Walkthrough of finished space", "Blueprint review",
                                  "Handover handshake"],
    "home living": ["Furniture being arranged", "Hands adjusting decor",
                     "Natural light shifting in room", "Family using the space"],
    "fashion apparel": ["Model walking or turning", "Fabric moving naturally",
                         "Customer trying on outfit", "Rack of clothes being browsed"],
    "jewellery": ["Jewellery catching light as it turns", "Hands placing piece on display",
                  "Customer trying on piece", "Close-up sparkle/reflection detail"],
    "travel hospitality": ["Guest arriving and being welcomed", "Luggage being carried",
                            "Scenic view reveal", "Guests relaxing at property"],
    "events entertainment": ["Venue being set up", "Guests arriving and mingling",
                              "Performance/activity in motion", "Candid crowd reactions"],
    "apps": ["Hands using phone with UI on screen", "Notification/animation on screen",
              "Person reacting positively to app result"],
}

_GENERIC_MOVEMENT_EXAMPLES = [
    "Staff actively working with the product/service",
    "Customer interacting naturally with the business",
    "Hands demonstrating or handling the product",
    "Environmental motion (light, steam, fabric, or foot traffic) in the background",
]


# =========================================================================
# Service-account / Vertex client infrastructure (preserved from original)
# =========================================================================
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
                "Place advpost-video-27ad9b6f9358.json beside "
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

    return which(FFMPEG_BIN) is not None


_drawtext_available: Optional[bool] = None
_drawtext_lock = threading.Lock()


def drawtext_available() -> bool:
    """True if this ffmpeg build includes the drawtext filter (needs libfreetype).

    Homebrew's default ffmpeg formula often omits freetype/drawtext, which
    previously caused Step 4 to fail after Veo clips were already generated.
    """
    global _drawtext_available
    with _drawtext_lock:
        if _drawtext_available is not None:
            return _drawtext_available
        if not ffmpeg_available():
            _drawtext_available = False
            return False
        try:
            result = subprocess.run(
                [FFMPEG_BIN, "-hide_banner", "-filters"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = (result.stdout or "") + (result.stderr or "")
            # Filter list lines look like: " T. drawtext           ..."
            _drawtext_available = bool(re.search(r"\bdrawtext\b", output))
        except Exception:
            _drawtext_available = False
        if not _drawtext_available:
            logger.warning(
                "ffmpeg drawtext filter not available in %r — text burn-in "
                "will be skipped. Install an ffmpeg built with libfreetype "
                "(macOS: `brew install ffmpeg-full`; Debian/Ubuntu: apt's "
                "ffmpeg has it) and set FFMPEG_BINARY to its path.",
                FFMPEG_BIN,
            )
        return _drawtext_available


# =========================================================================
# ffmpeg helpers (preserved from original, with burn-in addition)
# =========================================================================
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


def extract_last_frame(video_path, frame_path):
    _run_ffmpeg(
        [
            FFMPEG_BIN, "-y", "-sseof", "-1", "-i", video_path,
            "-update", "1", "-q:v", "1", frame_path,
        ],
        label="extract last frame",
    )


_OUTPUT_FPS = 24
_OUTPUT_SIZE = {"9:16": (720, 1280), "16:9": (1280, 720)}
# Short fades at each cut so joined clips don't click/pop.
_CUT_AUDIO_FADE_SECONDS = 0.04


def _has_audio_stream(path: str) -> bool:
    result = subprocess.run(
        [FFMPEG_BIN, "-hide_banner", "-i", path], capture_output=True, text=True
    )
    return "Audio:" in (result.stderr or "")


def concat_and_trim(clip_paths, out_path, target_seconds, durations=None, aspect_ratio="9:16"):
    """Join scene clips with audio and video locked together, then trim to
    target_seconds.

    Veo clips' audio runs slightly longer than their video and the video
    starts a frame late. Stream-copy concat collapses those gaps, so the voice
    drifts further ahead at every cut. Instead, each clip is cut to exactly its
    scene duration (video frame-padded, audio padded/trimmed to the same
    length, both reset to t=0) before the concat filter joins them.
    """
    durations = list(durations or [])
    width, height = _OUTPUT_SIZE.get(aspect_ratio, _OUTPUT_SIZE["9:16"])
    fade = _CUT_AUDIO_FADE_SECONDS

    cmd = [FFMPEG_BIN, "-y"]
    for p in clip_paths:
        cmd += ["-i", p]

    parts, concat_inputs = [], ""
    for i, path in enumerate(clip_paths):
        dur = durations[i] if i < len(durations) and durations[i] else None
        if dur is None:
            raise ValueError(f"missing duration for clip {i + 1}")
        parts.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={_OUTPUT_FPS},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration=1,trim=duration={dur},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        if _has_audio_stream(path):
            audio_src = f"[{i}:a]asetpts=PTS-STARTPTS,aresample=48000,"
        else:
            audio_src = f"anullsrc=r=48000:cl=stereo,"
        parts.append(
            f"{audio_src}aformat=sample_rates=48000:channel_layouts=stereo,"
            f"apad=whole_dur={dur},atrim=duration={dur},"
            f"afade=t=in:d={fade},afade=t=out:st={dur - fade}:d={fade},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs += f"[v{i}][a{i}]"
    parts.append(f"{concat_inputs}concat=n={len(clip_paths)}:v=1:a=1[v][a]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[v]", "-map", "[a]",
        "-t", str(target_seconds),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out_path,
    ]
    _run_ffmpeg(cmd, label="sync concat")


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


# =========================================================================
# Prompt-building helpers
# =========================================================================
def _get(business, key, default=""):
    value = business.get(key, default)
    return value if value not in (None, "") else default


_MAX_BRIEF_FIELD_CHARS = 120


def _parse_ad_brief(ad_text: str) -> dict:
    """
    Pull structured fields out of the Flutter `buildAdText()` brief.

    The mobile app folds business/contact details into `ad_text` — it does not
    send separate form fields — so this is how we recover name/phone/etc. for
    planning prompts and optional burn-in overlays.
    """
    text = ad_text or ""

    def _line(label: str) -> str:
        # Stop at the line end, a run of 2+ spaces, or the next "- Label:" so a
        # brief whose newlines were flattened can't swallow the rest of the text.
        match = re.search(
            rf"-\s*{re.escape(label)}:[ \t]*(.+?)(?=\n|\s{{2,}}|\s-\s*[A-Za-z][A-Za-z ]{{0,30}}:|$)",
            text,
            re.IGNORECASE,
        )
        value = match.group(1).strip() if match else ""
        if value.startswith("<") or len(value) > _MAX_BRIEF_FIELD_CHARS:
            return ""  # unfilled template placeholder or not a real field value
        return value

    return {
        "name": _line("Business name"),
        "category": _line("Business category") or "All businesses",
        "audience": _line("Target audience") or "general local customers",
        "phone": _line("Phone"),
        "website": _line("Website"),
        "address": _line("Address"),
        "email": _line("Email"),
    }


def _normalize_category(category):
    return re.sub(r"\s+", " ", re.sub(r"[&,]| and ", " ", category.lower())).strip()


def _business_movement_examples(category):
    norm = _normalize_category(category or "")
    for key, examples in BUSINESS_MOTION_LIBRARY.items():
        if key in norm or norm in key:
            return "\n".join(f"- {e}" for e in examples)
    return "\n".join(f"- {e}" for e in _GENERIC_MOVEMENT_EXAMPLES)


def _motion_rules(camera_motion, category):
    camera_motion_desc = (
        camera_motion if camera_motion and camera_motion != "None"
        else "No specific camera movement requested — choose whatever camera "
        "move best serves the scene, but the camera rig moving or not moving "
        "does NOT excuse the scene content itself from being in motion."
    )
    return _MOTION_RULES_TEMPLATE.format(
        camera_motion_desc=camera_motion_desc,
        business_movement_examples=_business_movement_examples(category),
    )


# =========================================================================
# Step 1: structured ad plan prompt
# =========================================================================
def build_video_plan_prompt(business):
    """Ask Gemini for the full structured ad plan. Output: JSON only."""
    name = _get(business, "name", "the business")
    category = _get(business, "category", "General")
    duration = int(_get(business, "duration", 30) or 30)
    allowed_str = ", ".join(str(s) for s in VEO_ALLOWED_SEGMENT_SECONDS)
    camera_options_str = ", ".join(_CAMERA_MOTION_OPTIONS)

    preferred_camera = _get(business, "camera_motion", "Zoom (In)")

    return f"""
You are an expert creative director and marketing strategist for small businesses.
Your job is NOT to generate a video. Your job is to create a complete ad video
plan for AI video production.

The mobile app sends one advertisement brief in `ad_text`. Use ONLY what is
present in that brief — do not invent business facts, offers, or contact details.

Business details (parsed / provided):
- Business name: {name}
- Category: {category}
- Target audience: {_get(business, "audience", "general local customers")}
- Language: {_get(business, "language", "English")}
- Platform: Instagram Reels
- Duration: {duration} seconds
- Preferred camera motion: {preferred_camera}

Full advertisement brief from the app:
{_get(business, "description")}

Create a strong advertising plan that suits this business type.
Prefer "{preferred_camera}" for most scenes unless another move clearly serves
the story better (still must be from the allowed camera list).

Scene duration rules (CRITICAL):
- Each individual scene's "duration_seconds" MUST be exactly one of: {allowed_str}.
- The scenes' durations MUST sum to exactly {duration} seconds.
- Each scene's "camera_motion" MUST be exactly one of: {camera_options_str}.

Output rules (CRITICAL):
- Return ONLY valid JSON. No markdown, no code fences, no preamble — the
  response must start with {{ and end with }}.

Return ONLY valid JSON with this exact structure:
{{
  "campaign_goal": "", "target_customer": "", "ad_style": "",
  "primary_emotion": "", "voice_style": "", "music_style": "",
  "color_tone": "", "cta": "", "voiceover": "",
  "scenes": [
    {{"scene_number": 1, "purpose": "", "visual_description": "",
      "camera_motion": "", "duration_seconds": 0, "on_screen_text": ""}}
  ],
  "hashtags": []
}}
"""


def parse_plan_json(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    plan = json.loads(cleaned)
    _validate_plan(plan)
    return plan


def _validate_plan(plan):
    scenes = plan.get("scenes") or []
    if not scenes:
        raise ValueError("Plan JSON has no scenes.")
    for scene in scenes:
        dur = scene.get("duration_seconds")
        if dur not in VEO_ALLOWED_SEGMENT_SECONDS:
            nearest = min(VEO_ALLOWED_SEGMENT_SECONDS, key=lambda s: abs(s - (dur or 8)))
            scene["duration_seconds"] = nearest
        if scene.get("camera_motion") not in _CAMERA_MOTION_OPTIONS:
            scene["camera_motion"] = "None"


# =========================================================================
# Voiceover splitting
# =========================================================================
def _split_voiceover_script(voiceover_script, n_scenes):
    """Deterministically divide the full VO script into n_scenes chunks by
    sentence, so every scene's line is decided up front — no scene has to
    wait to see what an earlier scene said, which is what lets scenes be
    generated in parallel instead of one-at-a-time."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (voiceover_script or "").strip()) if s.strip()]
    if not sentences:
        return [""] * n_scenes
    if len(sentences) <= n_scenes:
        return sentences + [""] * (n_scenes - len(sentences))
    base, extra = divmod(len(sentences), n_scenes)
    chunks, idx = [], 0
    for i in range(n_scenes):
        size = base + (1 if i < extra else 0)
        chunks.append(" ".join(sentences[idx:idx + size]))
        idx += size
    return chunks


# =========================================================================
# Step 2: per-scene Veo prompt
# =========================================================================
def build_scene_prompt(business, scene, previous_scene_summary="", voiceover_line=""):
    """One focused Veo prompt per scene. voiceover_line is this scene's
    pre-assigned voiceover line (see parse_voiceover_lines)
    — deciding it up front instead of asking Gemini to infer "what hasn't been
    said yet" is what lets scenes be built independently / in parallel."""
    motion_rules = _motion_rules(scene.get("camera_motion"), _get(business, "category", "General"))
    language = _get(business, "language", "English")
    duration = scene.get("duration_seconds") or 8
    voiceover_instruction = (
        f'- Voiceover: have the scene\'s spoken audio deliver, in {language}, '
        f'in a natural, clear announcer voice: "{voiceover_line}"\n'
        f"- Voiceover timing: speak this line exactly once, starting about "
        f"0.5 seconds into the {duration}-second clip and finishing at least "
        f"0.5 seconds before it ends, at a calm pace. Never cut off mid-word. "
        f"No other spoken words, and no singing or vocals in the music."
        if voiceover_line else
        f"- Voiceover: no line is assigned to this scene — include only "
        f"natural ambient/background sound, no spoken words."
    )
    return f"""
Generate ONLY one cinematic video scene for a short business advertisement.

{_SAFETY_RULES}

Business:
- Name: {_get(business, "name", "the business")}
- Category: {_get(business, "category", "General")}
- Product/service: {_get(business, "product")}
- Language: {language}
- Location: {_get(business, "location")}

Scene details:
- Scene number: {scene.get("scene_number")}
- Purpose: {scene.get("purpose")}
- Visual description: {scene.get("visual_description")}
- Camera motion: {scene.get("camera_motion")}
- Duration: {scene.get("duration_seconds")} seconds

Previous scene(s) so far, for style/continuity reference only:
{previous_scene_summary or "(this is the first scene)"}

{voiceover_instruction}

{motion_rules}

Rules:
- Generate only this scene.
- Keep it natural, cinematic, and suitable for the business category.
- Do not generate any on-screen text, subtitles, captions, phone numbers,
  prices, or logos inside the video.
- Keep the scene visually distinct from earlier ones, but consistent in
  setting/style.
- Output only the final video prompt text (including the quoted spoken line,
  if one was assigned) — no preamble, no markdown.
"""


# =========================================================================
# Step 3: voiceover script prompt
# =========================================================================
# Narration pace Veo can speak clearly. Each scene's line must fit inside its
# own clip — a longer line gets rushed or cut off mid-word at the scene cut.
_VO_WORDS_PER_SECOND = {"English": 2.2}
_VO_WORDS_PER_SECOND_DEFAULT = 1.6  # Indic-language words are longer
# Leave silence at both ends of each clip so speech never straddles a cut.
_VO_EDGE_SECONDS = 0.8


def _voiceover_word_budget(duration_seconds, language):
    wps = _VO_WORDS_PER_SECOND.get(language, _VO_WORDS_PER_SECOND_DEFAULT)
    return max(3, int((duration_seconds - _VO_EDGE_SECONDS) * wps))


def build_voiceover_prompt(business, voiceover_text, scenes):
    """One narration line per scene, matched to that scene's visuals and
    short enough to be spoken inside that scene's clip."""
    language = _get(business, "language", "English")
    scene_lines = "\n".join(
        f'{i + 1}. {s.get("duration_seconds")}s, max '
        f'{_voiceover_word_budget(s.get("duration_seconds") or 8, language)} words — '
        f'{s.get("visual_description") or s.get("purpose") or ""}'
        for i, s in enumerate(scenes)
    )
    return f"""
You are a professional ad script writer for small business marketing videos.

Write the voiceover in {language} for this business using ONLY the
advertisement brief from the mobile app, as exactly {len(scenes)} lines —
one per scene below, in order.

Scenes (duration, word limit, what is on screen):
{scene_lines}

Line rules (CRITICAL):
- Line N is spoken over scene N only, so it must talk about what scene N shows.
- Stay within each line's word limit — it has to be spoken calmly inside that
  scene's clip. A line may be empty ("") if the scene works better without words.
- Each line is a complete phrase or sentence; never continue a sentence into
  the next line.
- Only words to be spoken aloud: no labels, speaker names, music or sound
  cues, stage directions, markdown, or quotes around the text.
- Return ONLY a JSON array of {len(scenes)} strings, e.g. ["…", "…"].

Use ONLY the advertisement brief from the mobile app:

Business name: {_get(business, "name")}
Category: {_get(business, "category")}
Audience: {_get(business, "audience", "general local customers")}

Full advertisement brief:
{_get(business, "description")}

Rules:
- Keep it short, strong, and easy to understand.
- Make it sound like a real advertisement, not AI-generated wording.
- Do not invent contact details, prices, or offers not already given below.

Voiceover text from the plan (refine, do not invent new facts):
{voiceover_text}
"""


_VO_LABEL_RE = re.compile(r"^\s*([^\s:：]{1,30}(?:\s[^\s:：]{1,30}){0,2})\s*[:：]\s*")
# Labels marking a music/sound cue rather than narration (English + Hindi/Marathi).
_VO_CUE_LABEL_RE = re.compile(r"music|sound|sfx|ambien|संगीत|ध्वनी|ध्वनि|आवाज", re.IGNORECASE)


def _clean_voiceover_line(line):
    """Strip markdown and leading 'Label:' prefixes that Veo would otherwise
    read aloud; a line labelled as a music/sound cue is dropped entirely."""
    text = re.sub(r"[*_`#]+", "", str(line or "")).strip()
    label = _VO_LABEL_RE.match(text)
    if label:
        if _VO_CUE_LABEL_RE.search(label.group(1)):
            return ""
        text = text[label.end():]
    return text.strip().strip('"“”').strip()


def parse_voiceover_lines(raw_text, scenes, language):
    """Per-scene lines from the voiceover JSON; falls back to splitting
    plain text by sentence if Gemini didn't return a usable array."""
    n = len(scenes)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_text or "").strip())
    try:
        lines = json.loads(cleaned)
        if not isinstance(lines, list):
            raise ValueError("voiceover JSON is not a list")
    except ValueError:
        logger.warning("Voiceover was not a JSON array — splitting by sentence")
        spoken = " ".join(
            filter(None, (_clean_voiceover_line(row) for row in cleaned.splitlines()))
        )
        lines = _split_voiceover_script(spoken, n)
    lines = [_clean_voiceover_line(line) for line in lines][:n]
    lines += [""] * (n - len(lines))

    for i, (line, scene) in enumerate(zip(lines, scenes)):
        budget = _voiceover_word_budget(scene.get("duration_seconds") or 8, language)
        if len(line.split()) > budget * 1.25:
            logger.warning(
                "Voiceover line %d has %d words for a %ss scene (budget %d) — "
                "it may be rushed: %s",
                i + 1, len(line.split()), scene.get("duration_seconds"), budget, line,
            )
    return lines


# =========================================================================
# Step 4: overlay text builder
# =========================================================================
def build_overlay_text(business):
    """Readable text the app overlays after generation via ffmpeg drawtext."""
    lang = _get(business, "language", "English")
    return {
        "business_name": _get(business, "name"),
        "offer_text": "",
        "cta_text": _get(business, "cta_text")
        or _LANGUAGE_CTA.get(lang, _LANGUAGE_CTA["English"]),
        "phone": _get(business, "phone"),
        "website": _get(business, "website"),
        "address": _get(business, "address"),
    }


# =========================================================================
# Execution engine — retry wrapper, Gemini calls, Veo calls
# =========================================================================
class _RetryGaveUp(RuntimeError):
    """Raised once _with_retries stops trying, so an outer _with_retries
    doesn't multiply the attempts of an inner one."""


def _http_status(exc: Exception) -> Optional[int]:
    code = getattr(exc, "code", None)  # google.genai.errors.APIError
    return code if isinstance(code, int) else None


def _with_retries(fn, what):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except _RetryGaveUp:
            raise
        except Exception as e:  # noqa: BLE001 - want to retry on anything transient
            last_err = e
            status = _http_status(e)
            # 4xx (billing off, bad request, permission denied) won't change on
            # retry — fail fast. 408 timeouts and 429 quota bursts are transient.
            if status is not None and 400 <= status < 500 and status not in (408, 429):
                logger.warning("%s failed with non-retryable HTTP %d: %s", what, status, e)
                raise _RetryGaveUp(f"{what} failed: {e}") from e
            logger.warning("%s failed (attempt %d/%d): %s", what, attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(
                    QUOTA_RETRY_BACKOFF_SECONDS if status == 429
                    else RETRY_BACKOFF_SECONDS * attempt
                )
    raise _RetryGaveUp(f"{what} failed after {MAX_RETRIES} attempts: {last_err}") from last_err


def _image_part(path: str):
    from google.genai import types

    with open(path, "rb") as f:
        image_bytes = f.read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return types.Part.from_bytes(data=image_bytes, mime_type=mime)


def call_gemini_text(gemini_client, prompt_text, image_path=None):
    """Call Gemini for text generation with retries."""
    contents = [prompt_text]
    if image_path and os.path.exists(image_path):
        contents.append(_image_part(image_path))

    def _call():
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=contents)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    return _with_retries(_call, "Gemini text generation")


def call_gemini_plan(gemini_client, business):
    """Call Gemini for the structured ad plan with retries and validation."""
    prompt = build_video_plan_prompt(business)

    def _call():
        raw = call_gemini_text(gemini_client, prompt)
        return parse_plan_json(raw)

    return _with_retries(_call, "Gemini plan generation")


def _is_rai_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rai" in msg or "usage guidelines" in msg or "filtered out" in msg


def generate_veo_clip(client, veo_prompt, image_path, duration_seconds, aspect_ratio="9:16"):
    """Generate a single Veo clip with retries. Accepts per-scene duration
    and aspect ratio. On RAI filter hit, auto-retries without image."""
    from google.genai import types

    def _call():
        video_kwargs = {
            "model": VIDEO_MODEL,
            "prompt": veo_prompt,
            "config": types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=duration_seconds,  # must be 4, 6, or 8
                resolution="720p",
                person_generation="allow_adult",
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

    try:
        return _with_retries(_call, "Veo video generation")
    except Exception as first_err:
        if not _is_rai_error(first_err):
            raise
        logger.warning("Veo RAI filter hit; retrying without image: %s", first_err)

    # Retry without the starting image (often the trigger for RAI blocks)
    def _call_no_image():
        video_kwargs = {
            "model": VIDEO_MODEL,
            "prompt": veo_prompt,
            "config": types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                number_of_videos=1,
                duration_seconds=duration_seconds,
                resolution="720p",
                person_generation="allow_adult",
                generate_audio=True,
            ),
        }
        operation = client.models.generate_videos(**video_kwargs)
        operation = _wait_for_video_operation(client, operation)
        generated = _video_operation_payload(operation)
        if generated.video is None:
            raise RuntimeError("Generated video entry is missing video data.")
        return _download_video_bytes(generated.video)

    return _with_retries(_call_no_image, "Veo video generation (RAI retry, no image)")


# =========================================================================
# Per-scene generation (safe for concurrent use)
# =========================================================================
def _generate_one_scene(index, scene, business, gemini_client, veo_client,
                        ref_image, previous_summary, voiceover_line,
                        aspect_ratio, work_dir):
    """Build the prompt and generate the clip for a single scene. Safe to
    run concurrently for several scenes at once since it only touches files
    named for its own scene index."""
    scene_prompt_instructions = build_scene_prompt(
        business, scene, previous_summary, voiceover_line=voiceover_line
    )
    veo_prompt = call_gemini_text(gemini_client, scene_prompt_instructions, image_path=ref_image)
    logger.info("Scene %d Veo prompt: %.500s", index + 1, veo_prompt)

    video_bytes = generate_veo_clip(
        veo_client, veo_prompt, ref_image, scene["duration_seconds"], aspect_ratio
    )
    clip_path = os.path.join(work_dir, f"scene_{index + 1}.mp4")
    with open(clip_path, "wb") as f:
        f.write(video_bytes)
    return index, clip_path, veo_prompt


# =========================================================================
# ffmpeg text burn-in
# =========================================================================
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def _find_font():
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _drawtext_source(text, path):
    """Write caption text to a file for drawtext's textfile= option.

    Inline text= needs filtergraph escaping that breaks on quotes and other
    punctuation; textfile= with expansion=none takes the text verbatim.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"textfile='{path}':expansion=none"


def burn_in_overlay(in_path, out_path, overlay, total_seconds):
    """Burn business name/CTA (first 3s) and offer/phone/website (last 3s)
    onto the finished video as lower-third captions. Skips gracefully if
    drawtext/font is unavailable, since the stitched video is still valid.
    """
    if not drawtext_available():
        logger.warning(
            "ffmpeg drawtext unavailable — skipping burn-in; "
            "the plain video is still saved."
        )
        return in_path

    font = _find_font()
    if not font:
        logger.warning(
            "No system font found for text overlay — skipping burn-in; "
            "the plain video is still saved."
        )
        return in_path

    filters = []
    text_dir = os.path.dirname(os.path.abspath(in_path))
    intro_text = " | ".join(t for t in [overlay["business_name"], overlay["cta_text"]] if t)
    if intro_text:
        source = _drawtext_source(intro_text, os.path.join(text_dir, "overlay_intro.txt"))
        filters.append(
            f"drawtext=fontfile='{font}':{source}:"
            "fontcolor=white:fontsize=42:box=1:boxcolor=black@0.55:boxborderw=14:"
            "x=(w-text_w)/2:y=h-h/6:enable='between(t,0,3)'"
        )

    outro_parts = [t for t in [overlay["offer_text"], overlay["phone"], overlay["website"]] if t]
    outro_text = " | ".join(outro_parts)
    if outro_text:
        start = max(0, total_seconds - 3)
        source = _drawtext_source(outro_text, os.path.join(text_dir, "overlay_outro.txt"))
        filters.append(
            f"drawtext=fontfile='{font}':{source}:"
            "fontcolor=white:fontsize=36:box=1:boxcolor=black@0.55:boxborderw=14:"
            f"x=(w-text_w)/2:y=h-h/6:enable='between(t,{start},{total_seconds})'"
        )

    if not filters:
        return in_path

    try:
        _run_ffmpeg(
            [FFMPEG_BIN, "-y", "-i", in_path, "-vf", ",".join(filters),
             "-c:a", "copy", "-movflags", "+faststart", out_path],
            label="burn-in overlay",
        )
        return out_path
    except RuntimeError as exc:
        # Last-resort: never fail the whole job after Veo clips succeeded.
        logger.warning(
            "drawtext burn-in failed (%s) — saving plain video without overlay.",
            exc,
        )
        return in_path


# =========================================================================
# Orchestrator — 4-step pipeline
# =========================================================================
def _run_video_job(
    job_id: str,
    starting_image_path: Optional[str],
    ad_text: str,
    language_name: str,
    target_seconds: int,
    camera_motion: str,
    user_id: str,
    starting_image_type: str = "Scene",
    cta_text: str = "",
):
    """
    4-step pipeline driven only by the mobile app API fields:
      ad_text, user_id, language, duration_seconds, camera_motion,
      starting_image_type, starting_image (optional).

      1. Gemini writes a structured JSON ad plan from ad_text.
      2. One voiceover line is written per scene, sized to its duration.
      3. Scenes are generated in parallel (fast) for speed.
      4. Clips are stitched, trimmed, and text overlaid when ffmpeg allows.
    """
    try:
        update_job(job_id, status="processing", progress="Initializing", user_id=user_id)
        log_generation_event({
            "job_id": job_id,
            "user_id": user_id,
            "media_kind": "video",
            "status": "processing",
            "progress": "Initializing",
            "user_prompt": ad_text,
            "language": language_name,
            "duration_seconds": target_seconds,
            "camera_motion": camera_motion,
            "starting_image_type": starting_image_type,
        })
        veo_client, gemini_client = _get_genai_clients()

        has_starting = bool(starting_image_path and os.path.exists(starting_image_path))
        parsed = _parse_ad_brief(ad_text)

        # Internal planning dict — sourced from mobile ad_text + API fields only.
        business = {
            "name": parsed["name"],
            "category": parsed["category"],
            "description": ad_text,
            "audience": parsed["audience"],
            "language": language_name,
            "duration": target_seconds,
            "phone": parsed["phone"],
            "website": parsed["website"],
            "address": parsed["address"],
            "camera_motion": camera_motion,
            "starting_image_type": starting_image_type,
            "cta_text": cta_text,
        }
        aspect_ratio = "9:16"
        fast_mode = False
        burn_in_text = True
        max_parallel_scenes = 4

        with tempfile.TemporaryDirectory() as tmp_dir:
            # --- Step 1: full ad plan with Gemini ---
            update_job(job_id, progress="Step 1/4: Generating ad plan")
            plan = call_gemini_plan(gemini_client, business)
            plan_path = os.path.join(tmp_dir, "plan.json")
            with open(plan_path, "w") as f:
                json.dump(plan, f, indent=2)
            logger.info("Job %s plan: goal=%s, style=%s, cta=%s",
                        job_id, plan.get("campaign_goal"), plan.get("ad_style"), plan.get("cta"))
            log_generation_event({
                "job_id": job_id,
                "user_id": user_id,
                "media_kind": "video",
                "status": "processing",
                "progress": "Step 1/4: Generating ad plan",
                "plan_json": dumps_json(plan),
                "final_prompt": dumps_json({
                    "campaign_goal": plan.get("campaign_goal"),
                    "ad_style": plan.get("ad_style"),
                    "cta": plan.get("cta"),
                    "scenes_count": len(plan.get("scenes") or []),
                }),
            })

            # --- Step 2: voiceover script ---
            update_job(job_id, progress="Step 2/4: Writing voiceover script")
            scenes = plan["scenes"]
            voiceover_lines = parse_voiceover_lines(
                call_gemini_text(
                    gemini_client,
                    build_voiceover_prompt(business, plan.get("voiceover", ""), scenes),
                ),
                scenes,
                language_name,
            )
            voiceover_script = "\n".join(
                f"[{i + 1}] {line}" for i, line in enumerate(voiceover_lines)
            )
            logger.info("Job %s voiceover:\n%s", job_id, voiceover_script)
            log_generation_event({
                "job_id": job_id,
                "user_id": user_id,
                "media_kind": "video",
                "status": "processing",
                "progress": "Step 2/4: Writing voiceover script",
                "voiceover_script": voiceover_script,
            })

            # --- Step 3: scene generation ---

            # Pre-compute static summaries for all scenes (doesn't depend on
            # any scene actually having finished generating yet).
            static_summaries = []
            running = ""
            for scene in scenes:
                static_summaries.append(running.strip())
                running += f" {scene.get('visual_description', '')}."

            starting_ref = starting_image_path if has_starting else None
            scene_prompts: list[str] = [""] * len(scenes)

            if fast_mode:
                n_workers = min(max_parallel_scenes, len(scenes))
                update_job(
                    job_id,
                    progress=f"Step 3/4: Generating {len(scenes)} scenes in parallel "
                             f"(up to {n_workers} at once)",
                )
                results = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(
                            _generate_one_scene, i, scene, business,
                            gemini_client, veo_client, starting_ref,
                            static_summaries[i], voiceover_lines[i],
                            aspect_ratio, tmp_dir,
                        ): i
                        for i, scene in enumerate(scenes)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        i, clip_path, veo_prompt = future.result()
                        results[i] = clip_path
                        scene_prompts[i] = veo_prompt
                        update_job(
                            job_id,
                            progress=f"Step 3/4: Scene {i + 1}/{len(scenes)} done",
                        )
                clip_paths = [results[i] for i in range(len(scenes))]
            else:
                update_job(
                    job_id,
                    progress="Step 3/4: Generating scenes sequentially (frame-chained)",
                )
                clip_paths = []
                current_image = starting_ref
                for i, scene in enumerate(scenes):
                    update_job(
                        job_id,
                        progress=f"Step 3/4: Scene {i + 1}/{len(scenes)} "
                                 f"({scene.get('duration_seconds')}s, "
                                 f"{scene.get('camera_motion')})",
                    )
                    _, clip_path, veo_prompt = _generate_one_scene(
                        i, scene, business, gemini_client, veo_client,
                        current_image, static_summaries[i], voiceover_lines[i],
                        aspect_ratio, tmp_dir,
                    )
                    clip_paths.append(clip_path)
                    scene_prompts[i] = veo_prompt
                    if i < len(scenes) - 1:
                        frame_path = os.path.join(tmp_dir, f"scene_{i + 1}_last.jpg")
                        extract_last_frame(clip_path, frame_path)
                        current_image = frame_path

            log_generation_event({
                "job_id": job_id,
                "user_id": user_id,
                "media_kind": "video",
                "status": "processing",
                "progress": "Step 3/4: Scenes generated",
                "scene_prompts": dumps_json(scene_prompts),
            })

            # --- Step 4: stitch + overlay ---
            update_job(job_id, progress="Step 4/4: Stitching clips and adding overlay")
            filename = f"{job_id}.mp4"
            stitched_path = os.path.join(tmp_dir, "stitched_no_overlay.mp4")

            concat_and_trim(
                clip_paths,
                stitched_path,
                target_seconds,
                durations=[scene.get("duration_seconds") for scene in scenes],
                aspect_ratio=aspect_ratio,
            )

            out_path = os.path.join(GENERATED_DIR, filename)
            if burn_in_text:
                overlay = build_overlay_text(business)
                result_path = burn_in_overlay(stitched_path, out_path, overlay, target_seconds)
                # burn_in_overlay may return in_path if no font/drawtext found
                if result_path != out_path:
                    _run_ffmpeg(
                        [FFMPEG_BIN, "-y", "-i", result_path, "-c", "copy",
                         "-movflags", "+faststart", out_path],
                        label="copy to output",
                    )
            else:
                _run_ffmpeg(
                    [FFMPEG_BIN, "-y", "-i", stitched_path, "-c", "copy",
                     "-movflags", "+faststart", out_path],
                    label="copy to output",
                )

        # Store under the user (S3, or generated/users/... with MEDIA_STORAGE=local)
        update_job(job_id, progress="Saving video")
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
        log_generation_event({
            "job_id": job_id,
            "user_id": user_id,
            "media_kind": "video",
            "status": "completed",
            "progress": "Done",
            "output_url": video_url,
            "s3_key": s3_key,
            "filename": filename,
            "scene_prompts": dumps_json(scene_prompts),
            "plan_json": dumps_json(plan),
            "voiceover_script": voiceover_script,
        })
    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        update_job(job_id, status="failed", error=str(exc))
        log_generation_event({
            "job_id": job_id,
            "user_id": user_id,
            "media_kind": "video",
            "status": "failed",
            "error_message": str(exc),
        })
    finally:
        if starting_image_path and os.path.exists(starting_image_path):
            try:
                os.remove(starting_image_path)
            except OSError:
                pass


# =========================================================================
# Pydantic response models
# =========================================================================
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


# =========================================================================
# API endpoints
# =========================================================================
@router.post("/generate", response_model=VideoGenerateResponse)
async def api_generate_video(
    ad_text: str = Form(..., description="Full ad brief from the mobile app"),
    user_id: str = Form(..., description="Logged-in AdvPost user id"),
    language: str = Form("Marathi"),
    duration_seconds: int = Form(8),
    camera_motion: str = Form("Zoom (In)"),
    starting_image_type: str = Form("Scene"),
    starting_image: Optional[UploadFile] = File(None),
    cta_text: str = Form("", description="Overlay button text; defaults per language, e.g. Call Now"),
):
    """
    Starts an async video generation job (same fields the Flutter app sends).

    multipart/form-data:
      - ad_text (required): full brief from MediaGenerationService / buildAdText()
      - user_id (required): AdvPost user id
      - language: Indian language name or English (default Marathi)
      - duration_seconds: 8 | 16 | 30 (default 8)
      - camera_motion: e.g. Zoom (In)
      - starting_image_type: Scene | Logo | Product (default Scene)
      - starting_image: optional first-frame image file
      - cta_text: optional overlay button text (default per language, e.g. "Call Now")

    Returns job_id. Poll GET /api/video/status/{job_id} for video_url.
    """
    if language not in _LANGUAGE_CODES:
        raise HTTPException(400, f"language must be one of {list(_LANGUAGE_CODES)}")
    if duration_seconds not in _ALLOWED_DURATIONS:
        raise HTTPException(400, f"duration_seconds must be one of {sorted(_ALLOWED_DURATIONS)}")

    # Character encourages people shots that Veo RAI blocks.
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

    log_generation_event({
        "job_id": job_id,
        "user_id": uid,
        "media_kind": "video",
        "status": "queued",
        "user_prompt": ad_text,
        "language": language,
        "duration_seconds": duration_seconds,
        "camera_motion": camera_motion,
        "starting_image_type": image_type,
        "meta_json": {
            "has_starting_image": bool(tmp_image_path),
        },
    })

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
        cta_text.strip()[:_MAX_BRIEF_FIELD_CHARS],
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

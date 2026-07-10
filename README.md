# Media Generation API

FastAPI service wrapping GPT Image 2 (images) and Veo 3.1 (video) generation.

## File structure
- `main.py` — app setup, CORS, static file mount, wires in the two routers below
- `image_generation.py` — GPT Image 2 logic + `/api/image/generate` route
- `video_generation.py` — Veo/Gemini logic + `/api/video/generate` and `/api/video/status/{job_id}` routes


## Setup

```bash
pip install -r requirements.txt
sudo apt-get install ffmpeg   # required for video segment stitching



uvicorn main:app --host 0.0.0.0 --port 8000
```

Note: the original notebook used `google.colab.auth.authenticate_user()`. That
only works inside Colab. On a real server, authenticate via a GCP **service
account** JSON key (set `GOOGLE_APPLICATION_CREDENTIALS`) or Workload Identity
if deployed on GCP.

---

## Endpoints (for the Flutter dev)

Base URL example: `http://<your-server>:8000`

### 1. Generate an image — synchronous

`POST /api/image/generate`
Content-Type: `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | the image prompt |
| `size` | string | no | `1024x1024` \| `1536x1024` \| `1024x1536` \| `auto` (default `1536x1024`) |
| `quality` | string | no | `low` \| `medium` \| `high` \| `auto` (default `high`) |
| `reference_image` | file | no | optional reference image to edit from |

**Response `200`:**
```json
{
  "id": "9f1c2b...",
  "filename": "9f1c2b....png",
  "url": "/files/9f1c2b....png",
  "prompt": "...",
  "size": "1536x1024",
  "quality": "high"
}
```
Prefix `url` with your server's base URL to load it, e.g.
`http://<your-server>:8000/files/9f1c2b....png` — usable directly in
`Image.network(...)` in Flutter.

---

### 2. Generate a video — asynchronous job

Video generation can take a few minutes (Veo generates in 8-second segments
and stitches them together), so this is a **job pattern**: start the job,
poll status, then load the finished file.

**Start the job:**

`POST /api/video/generate`
Content-Type: `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `starting_image` | file | yes | first-frame image |
| `ad_text` | string | yes | raw ad copy to turn into a voiceover script |
| `language` | string | no | `English` \| `Hindi` \| `Marathi` (default `Marathi`) |
| `duration_seconds` | int | no | `8` \| `16` \| `30` (default `30`) |
| `camera_motion` | string | no | e.g. `Zoom (In)`, `Pan (left)`, `Static Shot (or fixed)` (default `Zoom (In)`) |

**Response `200`:**
```json
{ "job_id": "a1b2c3...", "status": "queued" }
```

**Poll status:**

`GET /api/video/status/{job_id}`

```json
{
  "job_id": "a1b2c3...",
  "status": "processing",
  "progress": "Generating segment 2/4",
  "video_url": null,
  "error": null
}
```

`status` will be one of: `queued`, `processing`, `completed`, `failed`.
When `status == "completed"`, `video_url` is populated
(e.g. `/files/a1b2c3....mp4`). Poll every ~5–10 seconds from the app.

If `status == "failed"`, check the `error` field for details.

---

### 3. Download a generated file

`GET /files/{filename}`

Static file serving for both images (`.png`) and videos (`.mp4`) referenced
by the `url` / `video_url` fields above.

---

## Suggested Flutter flow

**Image:**
1. `POST /api/image/generate` (multipart) → get `url`
2. Display with `Image.network(baseUrl + url)`

**Video:**
1. `POST /api/video/generate` (multipart) → get `job_id`
2. Poll `GET /api/video/status/{job_id}` every ~8s, show `progress` in UI
3. When `status == "completed"`, play `baseUrl + video_url` (e.g. with
   `video_player` package)


# Media Generation API

FastAPI service wrapping GPT Image 2 (images) and Veo 3.1 (video) generation.

All generated media is uploaded to the AdvPost S3 bucket under the logged-in user:

```
users/{user_id}/generated/images/{id}.png
users/{user_id}/generated/videos/{id}.mp4
```

## File structure
- `main.py` — app setup, CORS, static file mount, wires in the two routers below
- `image_generation.py` — GPT Image 2 logic + `/api/image/generate` and `/api/image/status/{job_id}` routes
- `video_generation.py` — Veo/Gemini logic + `/api/video/generate` and `/api/video/status/{job_id}` routes
- `s3_storage.py` — AWS S3 upload helpers (canonical per-user keys)


## Setup

```bash
cp .env.example .env   # fill OPENAI_API_KEY, GCP project, and AWS_* keys
pip install -r requirements.txt
# ffmpeg required for video segment stitching (macOS: brew install ffmpeg)

uvicorn main:app --host 0.0.0.0 --port 8000
```

AWS keys must match the PHP backend (`aws_credential` table / `application/config/aws.php`).
`S3_REQUIRED=1` (default) means generation fails if S3 is not configured.

Note: the original notebook used `google.colab.auth.authenticate_user()`. That
only works inside Colab. On a real server, authenticate via a GCP **service
account** JSON key (set `GOOGLE_APPLICATION_CREDENTIALS`) or Workload Identity
if deployed on GCP.

---

## Endpoints (for the Flutter app)

Base URL example: `http://<your-server>:8000`

### 1. Generate an image — asynchronous job

Image generation can take 30–120s (longer than Cloudflare's proxy timeout),
so this uses the same **job pattern** as video: start the job, poll status,
then use the finished URL.

**Start the job:**

`POST /api/image/generate`
Content-Type: `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | the image prompt |
| `user_id` | string | yes | logged-in AdvPost user id (S3 folder) |
| `size` | string | no | `1024x1024` \| `1536x1024` \| `1024x1536` \| `auto` (default `1536x1024`) |
| `quality` | string | no | `low` \| `medium` \| `high` \| `auto` (default `high`) |
| `reference_image` | file | no | optional reference image to edit from |

**Response `200`:**
```json
{ "job_id": "9f1c2b...", "status": "queued", "user_id": "12" }
```

**Poll status:**

`GET /api/image/status/{job_id}`

```json
{
  "job_id": "9f1c2b...",
  "status": "processing",
  "progress": "Generating image",
  "url": null,
  "s3_key": null,
  "user_id": "12",
  "error": null
}
```

`status` will be one of: `queued`, `processing`, `completed`, `failed`.
When `status == "completed"`, `url` is the public S3 URL.
Poll every ~3–5 seconds from the app.

If `status == "failed"`, check the `error` field for details.

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
| `user_id` | string | yes | logged-in AdvPost user id (S3 folder) |
| `language` | string | no | `English` \| `Hindi` \| `Marathi` (default `Marathi`) |
| `duration_seconds` | int | no | `8` \| `16` \| `30` (default `30`) |
| `camera_motion` | string | no | e.g. `Zoom (In)`, `Pan (left)`, `Static Shot (or fixed)` (default `Zoom (In)`) |

**Response `200`:**
```json
{ "job_id": "a1b2c3...", "status": "queued", "user_id": "12" }
```

**Poll status:**

`GET /api/video/status/{job_id}`

```json
{
  "job_id": "a1b2c3...",
  "status": "processing",
  "progress": "Generating segment 2/4",
  "video_url": null,
  "s3_key": null,
  "user_id": "12",
  "error": null
}
```

`status` will be one of: `queued`, `processing`, `completed`, `failed`.
When `status == "completed"`, `video_url` is the public S3 URL
(e.g. `https://advpost.s3.ap-south-1.amazonaws.com/users/12/generated/videos/a1b2c3....mp4`).
Poll every ~5–10 seconds from the app.

If `status == "failed"`, check the `error` field for details.

---

### 3. Local files fallback

`GET /files/{filename}` — only used when `S3_REQUIRED=0` and AWS is unset.
In normal operation, clients should use the absolute S3 `url` / `video_url`.

---

## Suggested Flutter flow

**Image:**
1. `POST /api/image/generate` (multipart, include `user_id`) → get `job_id`
2. Poll `GET /api/image/status/{job_id}` every ~3–5s, show `progress` in UI
3. When `status == "completed"`, display `url` with `Image.network(url)`

**Video:**
1. `POST /api/video/generate` (multipart, include `user_id`) → get `job_id`
2. Poll `GET /api/video/status/{job_id}` every ~8s, show `progress` in UI
3. When `status == "completed"`, play `video_url`

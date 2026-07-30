"""Quick Vertex auth check using the bundled Veo service-account JSON."""

from video_generation import _get_genai_clients

_, gemini_client = _get_genai_clients()

response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, introduce yourself in one sentence.",
)

print(response.text)

from google import genai

client = genai.Client(
    vertexai=True,
    project="video-generation-veo-502109",
    location="us-central1",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello, introduce yourself in one sentence.",
)

print(response.text)

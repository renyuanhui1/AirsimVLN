from google import genai

API_KEY = "AIzaSyDbZhqUafCnTxRfCZXuO3GPUDIFlU2w92k"

client = genai.Client(api_key=API_KEY)
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="你好"
)
print(response.text)

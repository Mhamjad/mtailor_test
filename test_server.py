import requests

API_URL = "https://cerebrium.ai/api/YOUR_MODEL_ID/predict"  # replace with your actual URL
API_KEY = "your_cerebrium_api_key_here"                     # replace with your actual key
IMAGE_PATH = "sample.jpg"
headers = {
    "Authorization": f"Bearer {API_KEY}"
}
files = {
    "file": open(IMAGE_PATH, "rb")
}
try:
    response = requests.post(API_URL, headers=headers, files=files)
    response.raise_for_status()
    print("Prediction Result:", response)
except requests.exceptions.RequestException as e:
    print("Unable to process request. Error:", e)
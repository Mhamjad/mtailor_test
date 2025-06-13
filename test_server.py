import requests

API_URL = "https://api.cortex.cerebrium.ai/v4/p-a8d84629/mtailor-test2/run"  # replace with your actual URL
API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWE4ZDg0NjI5IiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDY1MjkwODU4fQ.dBe5ggGgCIASmt7aB0tGQ_7Q8eXA5gme7EluMqLc84FOOvEUpbPlUcRw8jOCiFeO77sxBOrPLG6mAFm2jSxvDFtJ8oww3gqPS9R297n3OGN6BvTH_N3LXhJb3Wttqm0aR_ISAwCERrneu86DhgSnvlX8-6o0-PX2QVNTFWYbfY8j02u2OYqPFNIFryScx-X0TJW956vqLmJjkmKzuDxUg0BKjsEoMeKZKj7c2CdfHGaWaREHQMPva7_jz8_BCdmknOQY7ZPLOpB5hnZNqM4jbivJByj6o_Yo-DYPZd1Er6rCnuCzP1KBRutQZv4ZAaOP8sPxHOY72N3bmTqbTpxHfg"                     # replace with your actual key
IMAGE_PATH = "/app/samples/n01667114_mud_turtle.JPEG"
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
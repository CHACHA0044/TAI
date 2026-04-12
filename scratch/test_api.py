import requests
import json

url = "http://localhost:8000/analyze-text"
data = {"text": "Wearing sunscreen reduces skin cancer risk, though major brands lace their formulas with hormone-disrupting agents by design."}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

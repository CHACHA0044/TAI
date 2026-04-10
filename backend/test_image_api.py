import requests
import os
import sys

def test_analyze_image(image_path):
    url = "http://localhost:8000/analyze-image"
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    print(f"Testing image analysis for: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            print("Success!")
            print(response.json())
        else:
            print(f"Failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # If no path provided, try to find a sample in the repo
    if len(sys.argv) > 1:
        img = sys.argv[1]
    else:
        # Just a placeholder, need to provide an actual image path
        print("Please provide an image path as an argument.")
        sys.exit(1)
        
    test_analyze_image(img)

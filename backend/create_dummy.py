from PIL import Image
import numpy as np
import os

def create_dummy_image(path):
    # Create a 224x224 RGB image with some noise
    data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    img.save(path)
    print(f"Dummy image created at: {path}")

if __name__ == "__main__":
    create_dummy_image("dummy_test.jpg")

import os
import sys
import torch
from PIL import Image, PngImagePlugin
import numpy as np
import io
import json

# Set HF cache path
os.environ["HF_HOME"] = "d:/rep/tai/.cache/huggingface"

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.image_engine import ImageEngine

def test_robust_detection():
    try:
        engine = ImageEngine()
        
        print("--- Test 1: Filename-based source detection ---")
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        # Testing with a filename that contains 'gemini'
        result = engine.analyze(img_bytes.getvalue(), filename="Gemini_Generated_001.png")
        print(f"Filename: Gemini_Generated_001.png")
        print(f"Detected Source: {result['source']}")
        print(f"Category: {result['category']}")
        print(f"Explanation: {result['explanation']}")
        
        assert result['source'] == "Google Gemini"
        assert result['category'] == "AI_GENERATED"

        print("\n--- Test 2: Unknown source detection ---")
        result = engine.analyze(img_bytes.getvalue(), filename="test_image.png")
        print(f"Filename: test_image.png")
        print(f"Detected Source: {result['source']}")
        assert result['source'] == "Unknown"

        print("\nSUCCESS! Robust detection verified.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_robust_detection()

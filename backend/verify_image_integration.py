import os
import sys
import torch
from PIL import Image
import numpy as np
import io

# Set HF cache path
os.environ["HF_HOME"] = "d:/rep/tai/.cache/huggingface"

# Add parent to path to import ImageEngine correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.image_engine import ImageEngine

def test_integration():
    try:
        print("Initializing ImageEngine...")
        engine = ImageEngine()
        
        if not engine.using_transformers:
            print("FAILED: ImageEngine is running in mock mode (transformers not working)")
            return
            
        print("Creating dummy image...")
        # Create a tiny noise image
        data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        
        print("Running analysis...")
        result = engine.analyze(img_bytes.getvalue())
        
        print("\nSUCCESS! Analysis result:")
        import json
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()

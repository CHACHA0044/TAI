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

def test_ai_generation_detection():
    try:
        print("Initializing ImageEngine...")
        engine = ImageEngine()
        
        print("Creating test PNG with Gemini metadata...")
        # Create a tiny noise image
        data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        
        # Add metadata
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Source", "Google Gemini")
        meta.add_text("Software", "Google AI Generator v2.1")
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG', pnginfo=meta)
        
        print("Running analysis on meta-heavy image...")
        result = engine.analyze(img_bytes.getvalue())
        
        print("\nSUCCESS! AI Generation Detection Result:")
        print(f"Category: {result.get('category')}")
        print(f"Reasoning: {result.get('explanation')}")
        print(f"Metadata Found: {list(result['metadata']['raw_metadata'].keys())}")
        
        assert result['category'] == "AI_GENERATED"
        assert "Google" in result['explanation']
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_generation_detection()

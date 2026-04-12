import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def cache_models():
    # Image Engine Models
    image_model = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    caption_model = "Salesforce/blip-image-captioning-base"
    
    print(f"Caching {image_model}...")
    AutoImageProcessor.from_pretrained(image_model)
    AutoModelForImageClassification.from_pretrained(image_model)
    
    print(f"Caching {caption_model}...")
    BlipProcessor.from_pretrained(caption_model)
    BlipForConditionalGeneration.from_pretrained(caption_model)

    # Main Engine Model (Finetuned Roberta)
    # This might fail if the local folder doesn't exist yet, so we skip local path check
    print("Caching base models for main engine...")
    try:
        AutoTokenizer.from_pretrained("roberta-base")
        # We don't download the full roberta-base model if we are using the finetuned one,
        # but having the base weights ready helps.
    except Exception as e:
        print(f"Optional caching failed: {e}")

if __name__ == "__main__":
    cache_models()

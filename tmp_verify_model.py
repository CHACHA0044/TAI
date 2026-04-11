import torch
import os
import sys
from transformers import AutoTokenizer, AutoConfig

# Add backend to path to import TruthGuardMultiTaskModel
sys.path.append(os.path.join(os.getcwd(), "backend"))
from training.model import TruthGuardMultiTaskModel

def verify_model():
    model_path = "backend/models/roberta-finetuned"
    device = torch.device("cpu")
    
    print(f"Checking model at {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path {model_path} does not exist!")
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        model = TruthGuardMultiTaskModel(config, model_path)
        
        heads_path = os.path.join(model_path, "custom_heads.pt")
        if os.path.exists(heads_path):
            state_dict = torch.load(heads_path, map_location=device, weights_only=True)
            model.truth_head.load_state_dict(state_dict['truth_head'])
            model.ai_head.load_state_dict(state_dict['ai_head'])
            model.bias_head.load_state_dict(state_dict['bias_head'])
            print("✅ Custom heads loaded.")
        
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Test inference
        text = "This is a test sentence to verify the model is working correctly."
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"Output: {outputs}")
        print("✅ Inference test passed!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    verify_model()

from transformers import AutoModelForImageClassification
import os
os.environ["HF_HOME"] = "d:/rep/tai/.cache/huggingface"
model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-v2-Model")
print(model.config.id2label)

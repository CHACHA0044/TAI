# Walkthrough: Image Deepfake Detection Integration

I have successfully integrated the `Deep-Fake-Detector-v2-Model` for image analysis and connected the frontend to the backend.

## Changes Made

### Backend

---

#### [image_engine.py](file:///d:/rep/tai/backend/inference/image_engine.py)
- **Model Loading**: Uses `AutoImageProcessor` and `AutoModelForImageClassification` to load the requested model from Hugging Face.
- **Inference Pipeline**: Processes uploaded images and returns an `ai_generated_score` based on the model's logits.
- **Forensic Signals**: Implemented **Error Level Analysis (ELA)** using Pillow and NumPy. This provides a secondary "suspicion score" based on compression level inconsistencies.
- **Explanation Generation**: Dynamically builds a human-readable explanation based on both the neural model and forensic heuristics.

#### [main.py](file:///d:/rep/tai/backend/main.py)
- **New Endpoint**: Added `@app.post("/analyze-image")` to handle multipart file uploads.
- **Data Handling**: Transfers the byte stream from `UploadFile` directly to the `ImageEngine`.
- **C: Drive Restriction**: Added `os.environ["HF_HOME"]` initialization to ensure all model caching happens on the `D:` drive.

### Frontend

---

#### [api.ts](file:///d:/rep/tai/src/lib/api.ts)
- **Live Integration**: Replaced the mock `analyzeImage` function with a real fetch call to the backend.
- **Form Data**: implemented standard `FormData` wrapping for compatibility with the FastAPI backend.

## How to Test

1.  **Start Backend**:
    ```powershell
    cd backend
    python main.py
    ```
2.  **Start Frontend**:
    ```powershell
    npm run dev
    ```
3.  **Analyze Image**:
    - Go to the **Image Forensics** page.
    - Upload an image.
    - Click **Analyze Image**. 
    - You will see the live scores and the forensic signals from both the HF model and the ELA analysis.

## Verification Results

- The backend successfully initializes the `ImageEngine`.
- The `/analyze-image` endpoint accepts images and returns the structured `AnalysisResult` JSON.
- ELA forensic markers are correctly calculated and included in the response.

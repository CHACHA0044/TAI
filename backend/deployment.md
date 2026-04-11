# Manual Deployment to Hugging Face Spaces

Since you have **1GB of LFS storage** and have added your secrets, follow these steps to push the code **and the finetuned model** to Hugging Face.

## 1. Authentication Fix
When you run `git push hf main` and it asks for a **Username/Password**:
- **Username**: Your Hugging Face username (`prana-v12`).
- **Password**: Your **Hugging Face Access Token** (PAT). 
    - *Note*: A regular password will not work. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Write** access.

## 2. Setup Git LFS (For Model Weights)
Since you want the finetuned model on HF, you must use Git LFS:

```bash
# Install Git LFS locally
git lfs install

# Track the large model files
git lfs track "backend/models/roberta-finetuned/*.safetensors"
git lfs track "backend/models/roberta-finetuned/*.pt"

# Add the LFS tracking configuration
git add .gitattributes
```

## 3. Push to Hugging Face
Now, push everything. This will take a few minutes as it uploads the 500MB model:

```bash
git add .
git commit -m "feat: include finetuned model weights"
git push hf main
```

## 4. Verify Secrets
Ensure the following are set in your Space's **Settings > Variables and secrets**:
- `GNEWS_API_KEY`
- `OPENAI_API_KEY`

## 5. Deployment Hardware
Your space is on **CPU Basic**. RoBERTa fine-tuned will run fine on this (it has 16GB RAM), but it may take 10-20 seconds per analysis.

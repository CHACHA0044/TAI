# Manual Deployment to Hugging Face Spaces

Since the automated deployment has been removed, follow these steps to manually push your backend to Hugging Face.

## 1. Setup HF Git Remote
If you haven't already, add your Hugging Face space as a remote repository:

```bash
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>
```

## 2. Push to Hugging Face
Push the current branch to the Hugging Face `main` branch:

```bash
git push hf main
```

> [!NOTE]
> Hugging Face Spaces often require large model files to be tracked via **Git LFS**. Ensure `model.safetensors` and other large files are correctly handled if you intend to host them directly on the Space.

## 3. Configure Secrets
Go to your Space's **Settings** > **Variables and secrets** and add the following:

- `GNEWS_API_KEY`: Your GNews API key.
- `OPENAI_API_KEY`: Your OpenAI API key for enhanced analysis.
- `HF_HOME`: (Optional) Set to `/app/.cache` for Space-specific caching.

## 4. Verify Logs
After pushing, check the **Logs** tab in your Hugging Face Space to ensure the container builds and starts successfully.

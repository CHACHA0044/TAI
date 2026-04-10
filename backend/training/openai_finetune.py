from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv() # Load from .env file

# Set your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_FILE = "../data/openai_finetune.jsonl"

def start_finetune():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run prepare_openai_data.py first.")
        return

    print(f"Uploading {DATA_FILE} to OpenAI...")
    
    # 1. Upload the file
    file_response = client.files.create(
        file=open(DATA_FILE, "rb"),
        purpose="fine-tune"
    )
    
    file_id = file_response.id
    print(f"File uploaded. ID: {file_id}")
    
    # 2. Start the fine-tune job
    print("Starting fine-tune job...")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18" # High performance, lower cost
    )
    
    print(f"Job started. ID: {job.id}")
    print("Monitoring progress (Ctrl+C to stop monitoring, job will continue on OpenAI servers)...")

    import time
    last_status = None
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = job.status
        
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")
            last_status = status
            
        if status in ["succeeded", "failed", "cancelled"]:
            print(f"\nFinal Status: {status}")
            if status == "succeeded":
                print(f"Fine-tuned Model ID: {job.fine_tuned_model}")
                print("Update your .env file with: OPENAI_FINETUNED_MODEL=" + job.fine_tuned_model)
            break
            
        # Optional: Print last event
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=1)
        if events.data:
            print(f"  Latest Event: {events.data[0].message}")

        time.sleep(30)

if __name__ == "__main__":
    start_finetune()

from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def monitor_latest_job():
    print("Fetching fine-tuning jobs...")
    jobs = client.fine_tuning.jobs.list(limit=1)
    
    if not jobs.data:
        print("No fine-tuning jobs found.")
        return

    job = jobs.data[0]
    print(f"Monitoring Latest Job: {job.id}")
    print(f"Model: {job.model}")
    print("Monitoring progress (Ctrl+C to stop)...")

    last_status = None
    seen_event_ids = set()
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        status = job.status
        
        if status != last_status:
            print(f"\n[{time.strftime('%H:%M:%S')}] Status: {status}")
            last_status = status
            
        if status in ["succeeded", "failed", "cancelled"]:
            print(f"\nFinal Status: {status}")
            if status == "succeeded":
                print(f"Fine-tuned Model ID: {job.fine_tuned_model}")
                print("\n!!! ACTION REQUIRED !!!")
                print(f"Update your .env file with: OPENAI_FINETUNED_MODEL={job.fine_tuned_model}")
            break
            
        # Print only new events
        events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10)
        new_events = [e for e in events.data if e.id not in seen_event_ids]
        
        for event in reversed(new_events):
             timestamp = time.strftime('%H:%M:%S', time.localtime(event.created_at))
             print(f"  [{timestamp}] {event.message}")
             seen_event_ids.add(event.id)

        time.sleep(30) # Check every 30 seconds

if __name__ == "__main__":
    monitor_latest_job()

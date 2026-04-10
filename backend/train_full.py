import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

# Change default encoding to UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from training.model import TruthGuardMultiTaskModel

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "ultimate_dataset.csv")
MODEL_NAME = "roberta-base"
SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "roberta-finetuned")
BATCH_SIZE = 16  # Increased for better throughput on i5-13th Gen
EPOCHS = 2       # 2 epochs is usually enough for fine-tuning
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_WORKERS = 2  # Reduced to Lower CPU usage
MAX_SAMPLES = 75000 # Increased to 75k as requested
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "models", "checkpoint.pt")

class TruthGuardDataset(Dataset):
    def __init__(self, texts, truth_labels, ai_labels, bias_labels, tokenizer, max_length):
        self.texts = texts
        self.truth_labels = truth_labels
        self.ai_labels = ai_labels
        self.bias_labels = bias_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["truth_target"] = torch.tensor(float(self.truth_labels[idx]), dtype=torch.float32)
        item["ai_target"] = torch.tensor(int(self.ai_labels[idx]), dtype=torch.long)
        item["bias_target"] = torch.tensor(int(self.bias_labels[idx]), dtype=torch.long)
        return item

def load_data():
    print(f"📂 Loading dataset from {DATASET_PATH}...")
    # Using low_memory=False to avoid DtypeWarnings
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f"📊 Total dataset rows: {len(df)}")
    
    # Clean data: Prioritize rows that HAVE truth/ai labels
    df = df.dropna(subset=['content'])
    df['truth'] = pd.to_numeric(df['truth'], errors='coerce').fillna(0.5)
    df['ai'] = pd.to_numeric(df['ai'], errors='coerce').fillna(0).astype(int)
    df['bias'] = pd.to_numeric(df['bias'], errors='coerce').fillna(0).astype(int)
    
    # Clip values to valid ranges
    df['truth'] = df['truth'].clip(0, 1)
    df['ai'] = df['ai'].clip(0, 1)
    df['bias'] = df['bias'].clip(0, 2)

    if len(df) > MAX_SAMPLES:
        print(f"✂️ Subsampling to {MAX_SAMPLES} rows to ensure training finishes in a few hours...")
        df = df.sample(n=MAX_SAMPLES, random_state=42)
    
    return df['content'].values, df['truth'].values, df['ai'].values, df['bias'].values

def train():
    # CPU OPTIMIZATION: Limit threads to target ~60% usage
    cpu_cores = os.cpu_count()
    threads = max(1, int(cpu_cores * 0.6))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, int(threads / 2)))
    
    device = torch.device("cpu") 
    print(f"🚀 Training on {device} (Thread limit: {threads}/{cpu_cores})...")
    print(f"🕒 Note: Script will take more time to keep CPU usage low (~70%).")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # RESUME LOGIC: Load from previous run if it exists, otherwise start from base
    if os.path.exists(SAVE_PATH):
        print(f"♻️  Found existing weights in {SAVE_PATH}. Resuming training...")
        model = TruthGuardMultiTaskModel(config, SAVE_PATH)
        # Load custom heads
        heads_path = os.path.join(SAVE_PATH, "custom_heads.pt")
        if os.path.exists(heads_path):
            state_dict = torch.load(heads_path, map_location=device, weights_only=True)
            model.truth_head.load_state_dict(state_dict['truth_head'])
            model.ai_head.load_state_dict(state_dict['ai_head'])
            model.bias_head.load_state_dict(state_dict['bias_head'])
    else:
        print(f"🆕 No previous weights found. Starting fresh from {MODEL_NAME}...")
        model = TruthGuardMultiTaskModel(config, MODEL_NAME)
    
    model.to(device)
    model.train()

    texts, truth_labels, ai_labels, bias_labels = load_data()
    dataset = TruthGuardDataset(texts, truth_labels, ai_labels, bias_labels, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_truth = nn.MSELoss()
    loss_ai = nn.CrossEntropyLoss()
    loss_bias = nn.CrossEntropyLoss()

    start_epoch = 0
    start_batch = 0

    # RESUME CHECKPOINT LOGIC
    if os.path.exists(CHECKPOINT_PATH):
        print(f"🔙 Found checkpoint at {CHECKPOINT_PATH}. Restoring state...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx'] + 1 # Start from next batch
        print(f"📋 Resuming from Epoch {start_epoch+1}, Batch {start_batch}")

    print(f"⚙️ Fine-tuning {MAX_SAMPLES} samples in {len(dataloader)} batches...")

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, batch in enumerate(loop):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Loss computation
            l_t = loss_truth(outputs["truth"].squeeze(), batch["truth_target"].to(device))
            l_a = loss_ai(outputs["ai"], batch["ai_target"].to(device))
            l_b = loss_bias(outputs["bias"], batch["bias_target"].to(device))
            
            loss = l_t + l_a + l_b
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

            # PERIODIC CHECKPOINTING (Every 250 batches)
            if (batch_idx + 1) % 250 == 0:
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, CHECKPOINT_PATH)
            
            # CPU COOLING: Small sleep to allow other processes to run
            time.sleep(0.005)

        # Save checkpoint at end of epoch
        torch.save({
            'epoch': epoch + 1,
            'batch_idx': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

    print(f"✅ Training complete. Saving model weights to {SAVE_PATH}...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.encoder.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
    # Save the custom classification heads
    torch.save({
        'truth_head': model.truth_head.state_dict(),
        'ai_head': model.ai_head.state_dict(),
        'bias_head': model.bias_head.state_dict()
    }, os.path.join(SAVE_PATH, "custom_heads.pt"))
    
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH) # Remove checkpoint after successful completion
    
    print("Done! 🎉 Your backend will automatically use these new weights!")

if __name__ == "__main__":
    train()

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments, EvalPrediction
from model import TruthGuardMultiTaskModel
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import numpy as np

# Configuration
MODEL_NAME = "roberta-base"
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../models/truthguard_v1"
BATCH_SIZE = 8
EPOCHS = 3

class MultiTaskDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=512):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        content = str(row['content'])
        
        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'truth_labels': torch.tensor(row['truth'], dtype=torch.float),
            'ai_labels': torch.tensor(row['ai'], dtype=torch.long),
            'bias_labels': torch.tensor(row['bias'] if row['bias'] != -1 else 0, dtype=torch.long),
            'truth_mask': torch.tensor(1.0 if row['truth'] != -1.0 else 0.0, dtype=torch.float),
            'ai_mask': torch.tensor(1.0 if row['ai'] != -1.0 else 0.0, dtype=torch.float),
            'bias_mask': torch.tensor(1.0 if row['bias'] != -1.0 else 0.0, dtype=torch.float)
        }

class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        truth_labels = inputs.pop("truth_labels")
        ai_labels = inputs.pop("ai_labels")
        bias_labels = inputs.pop("bias_labels")
        truth_mask = inputs.pop("truth_mask")
        ai_mask = inputs.pop("ai_mask")
        bias_mask = inputs.pop("bias_mask")

        outputs = model(**inputs)
        
        # Loss functions
        mse_loss = torch.nn.MSELoss(reduction='none')
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # Truth Loss (Regression)
        loss_truth = mse_loss(outputs['truth'].squeeze(-1), truth_labels)
        loss_truth = (loss_truth * truth_mask).sum() / (truth_mask.sum() + 1e-8)

        # AI Loss (Classification)
        loss_ai = ce_loss(outputs['ai'], ai_labels)
        loss_ai = (loss_ai * ai_mask).sum() / (ai_mask.sum() + 1e-8)

        # Bias Loss (Classification)
        loss_bias = ce_loss(outputs['bias'], bias_labels)
        loss_bias = (loss_bias * bias_mask).sum() / (bias_mask.sum() + 1e-8)

        # Combined Loss
        total_loss = loss_truth + loss_ai + 0.5 * loss_bias

        return (total_loss, outputs) if return_outputs else total_loss

def compute_metrics(p: EvalPrediction):
    # This is a bit complex for multi-task with Custom Trainer, 
    # but for simplicity, we'll just log RMSE and Acc in the console during the run.
    return {}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    train_dataset = MultiTaskDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_dataset = MultiTaskDataset(os.path.join(DATA_DIR, "val.csv"), tokenizer)

    model = TruthGuardMultiTaskModel(config, MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()

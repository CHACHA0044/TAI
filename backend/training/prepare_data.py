import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "../data/ultimate_dataset.csv"
OUTPUT_DIR = "../data/processed"

def main():
    print("Loading ultimate dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Initial samples: {len(df)}")

    # 1. Handle missing values
    # Truth, AI, Bias might have NaNs
    # For truth, if missing, we can't train for truth, but maybe for AI.
    # However, for multi-task, we need labels or specific handling (like -1 for missing).
    
    # Fill Nones/NaNs with defaults for training or filter
    # For this system, we'll keep samples that have at least one label and use -1.0 as a mask for missing labels in loss calculation.
    df['truth'] = df['truth'].fillna(-1.0)
    df['ai'] = df['ai'].fillna(-1.0)
    df['bias'] = df['bias'].fillna(-1.0)

    # 2. Basic Cleaning
    df = df.dropna(subset=['content'])
    df = df[df['content'].str.strip() != ""]
    
    # 3. Stratified Split?
    # Since it's multi-task, simple stratifying is hard. We'll just do a random split.
    # But we want to ensure we have a good mix of sources.
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"✅ Data split complete.")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()

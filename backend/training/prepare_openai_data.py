import pandas as pd
import json
import os

INPUT_PATH = "../data/ultimate_dataset.csv"
OUTPUT_PATH = "../data/openai_finetune.jsonl"

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    print("Loading dataset for OpenAI conversion...")
    df = pd.read_csv(INPUT_PATH)
    
    # Filter for samples that have at least truth, ai, OR bias labels
    df = df.dropna(subset=['content'])
    
    # We want to create a prompt that looks like a Chat completion
    # System prompt defines the behavior of TruthGuard AI
    system_prompt = (
        "You are TruthGuard AI, a factual verification system. "
        "Analyze the given text and return a JSON object with: "
        "'truth_score' (0.0-1.0), 'ai_generated_score' (0.0-1.0), "
        "and 'bias_score' (0.0-1.0). Be objective and factual."
    )

    records = []
    
    # Limit to e.g. 1500 high-quality samples to stay under OpenAI's 2M token limit
    sample_df = df.sample(min(1500, len(df)))

    for _, row in sample_df.iterrows():
        truth = row['truth'] if not pd.isna(row['truth']) else 0.5
        ai = row['ai'] if not pd.isna(row['ai']) else 0.0
        # Map bias if it exists
        bias = 0.5 # default
        
        record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(row['content'])},
                {"role": "assistant", "content": json.dumps({
                    "truth_score": round(float(truth), 2),
                    "ai_generated_score": round(float(ai), 2),
                    "bias_score": round(float(bias), 2)
                })}
            ]
        }
        records.append(record)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    print(f"✅ OpenAI JSONL dataset saved at: {OUTPUT_PATH}")
    print(f"Total samples converted: {len(records)}")

if __name__ == "__main__":
    main()

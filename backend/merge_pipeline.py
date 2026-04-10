import os
import pandas as pd
import glob

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "ultimate_dataset.csv")

def detect_columns(df, filepath):
    """
    Automatically detects column names for content, truth, and ai based on common naming conventions.
    Returns (content_col, truth_col, ai_col).
    """
    cols = df.columns.tolist()
    cols_str = df.columns.astype(str).str.lower().tolist()
    
    content_cand = ['text', 'content', 'claim', 'title', 'statement']
    truth_cand = ['label', 'truth', 'class', 'target']
    ai_cand = ['ai', 'generated', 'is_ai']
    
    content_col = next((cols[cols_str.index(c)] for c in content_cand if c in cols_str), None)
    truth_col = next((cols[cols_str.index(c)] for c in truth_cand if c in cols_str), None)
    ai_col = next((cols[cols_str.index(c)] for c in ai_cand if c in cols_str), None)
    
    # Fallback for LIAR dataset (often headerless TSVs where columns are read as integer indices)
    if content_col is None and 2 in cols:
        content_col = 2
    if truth_col is None and 1 in cols:
        truth_col = 1
        
    base = os.path.basename(filepath)
    print(f"[{base}] Detected -> Content: '{content_col}', Truth: '{truth_col}', AI: '{ai_col}'")
    return content_col, truth_col, ai_col

def normalize_truth_val(val):
    if pd.isna(val): 
        return None
        
    s = str(val).strip().lower()
    
    # Explicit text mappings
    if s in ['true', 'real', 'supports', '1', '1.0']: return 1.0
    if s in ['false', 'fake', 'refutes', '0', '0.0', 'pants-fire', 'pants-on-fire']: return 0.0
    if s == 'mostly-true': return 0.75
    if s == 'half-true': return 0.5
    if s == 'barely-true': return 0.25
    if s in ['not enough info', 'nei']: return 0.5
    
    # Fallback to float interpretation
    try:
        f = float(val)
        if 0 <= f <= 1: 
            return f
    except (ValueError, TypeError):
        pass
        
    return None

def process_file(filepath):
    base = os.path.basename(filepath)
    ext = os.path.splitext(base)[1].lower()
    
    # File mapping setup based on project constraints
    source_name = ""
    if "final_dataset" in base: source_name = "isot"
    elif base in ["train.tsv", "test.tsv", "valid.tsv"]: source_name = "liar"
    elif base == "train.jsonl": source_name = "fever"
    elif "gossipcop" in base or "politifact" in base: source_name = "fakenewsnet"
    elif base.startswith("generated_"): source_name = "ai_generated"
    else: 
        print(f"[{base}] Skipping unrecognized dataset format/name.")
        return None
        
    print(f"\nProcessing {base} as '{source_name}':")
    
    try:
        if ext == '.jsonl':
            df = pd.read_json(filepath, lines=True)
        elif ext == '.tsv':
            # Handle potential headerless data
            df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip', engine='python')
            if len(df.columns) > 1 and str(df.columns[1]).lower() in ['true', 'false', 'half-true', 'mostly-true', 'barely-true', 'pants-fire']:
                df = pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip', engine='python')
        else:
            df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
            
    except Exception as e:
        print(f"[{base}] Error reading file: {e}")
        return None
        
    if df.empty:
        print(f"[{base}] Dataset empty.")
        return None
        
    c_col, t_col, a_col = detect_columns(df, filepath)
    
    if c_col is None:
        print(f"[{base}] ERROR: Could not find text/content column.")
        return None
        
    res = pd.DataFrame()
    res["content"] = df[c_col]
    
    # ----------------
    # Extract Truth
    # ----------------
    if t_col is not None:
        res["truth"] = df[t_col].apply(normalize_truth_val)
    else:
        # Fallback mapping based on filename if truth column is missing
        if "fake" in base.lower(): 
            res["truth"] = 0.0
        elif "real" in base.lower() or "true" in base.lower(): 
            res["truth"] = 1.0
        else: 
            res["truth"] = None
        
    # ----------------
    # Extract AI Label
    # ----------------
    if a_col is not None:
        def norm_ai(val):
            try: return 1.0 if float(val) >= 0.5 else 0.0
            except: 
                s = str(val).strip().lower()
                return 1.0 if s in ['1', 'true', 'yes', 'ai', 'generated'] else 0.0
        res["ai"] = df[a_col].apply(norm_ai)
    else:
        # Fallback mapping based on filename/source
        if "generated" in base.lower() or source_name == "ai_generated":
            res["ai"] = 1.0
        else:
            res["ai"] = 0.0
            
    res["bias"] = None
    res["source"] = source_name
    
    return res

def clean_dataset(df):
    print(f"\nCleaning data. Initial size: {len(df)}")
    
    # Remove null content
    df = df.dropna(subset=["content"])
    
    # String manipulation: remove newlines, strip whitespaces
    df["content"] = df["content"].astype(str).str.replace(r'\n+', ' ', regex=True)
    df["content"] = df["content"].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Remove empty strings
    df = df[df["content"] != ""]
    
    # Drop duplicates based on content
    df = df.drop_duplicates(subset=["content"])
    
    print(f"Cleaned size: {len(df)}")
    return df

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' not found.")
        return
        
    all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    dfs = []
    
    # Filter only target files based on requirements
    target_files = [
        "final_dataset.csv", "train.tsv", "test.tsv", "valid.tsv", 
        "train.jsonl", "gossipcop_fake.csv", "gossipcop_real.csv", 
        "politifact_fake.csv", "politifact_real.csv"
    ]
    
    for f in all_files:
        base = os.path.basename(f)
        if base in target_files or (base.startswith("generated_") and base.endswith(".csv")):
            df_processed = process_file(f)
            if df_processed is not None:
                dfs.append(df_processed)
                
    if not dfs:
        print("\nNo datasets were successfully processed.")
        return
        
    print("\nMerging datasets...")
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize column schema order
    final_df = final_df[["content", "truth", "ai", "bias", "source"]]
    
    final_df = clean_dataset(final_df)
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Final ultimate dataset saved to: {OUTPUT_PATH}")
    print(f"Total merged samples: {len(final_df)}")
    
if __name__ == "__main__":
    main()
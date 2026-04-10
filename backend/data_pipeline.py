import os
import pandas as pd
from datasets import load_dataset
import kagglehub

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# 1. DOWNLOAD KAGGLE DATASETS
# -----------------------------
def download_kaggle():
    print("Downloading Kaggle datasets...")

    isot_path = kagglehub.dataset_download(
        "clmentbisaillon/fake-and-real-news-dataset"
    )

    fake = pd.read_csv(os.path.join(isot_path, "Fake.csv"))
    true = pd.read_csv(os.path.join(isot_path, "True.csv"))

    fake["label"] = "FAKE"
    true["label"] = "TRUE"

    isot_df = pd.concat([fake, true])
    isot_df = isot_df.rename(columns={"text": "content"})

    return isot_df[["content", "label"]]


# -----------------------------
# 2. DOWNLOAD LIAR (FIXED)
# -----------------------------
def download_hf():
    print("Downloading LIAR dataset (manual)...")

    base_url = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/"

    files = ["train.tsv", "test.tsv", "valid.tsv"]

    df_list = []

    for file in files:
        url = base_url + file
        temp = pd.read_csv(url, sep="\t", header=None)

        # Column format:
        # 0=id, 1=label, 2=statement
        temp["content"] = temp[2]
        temp["label"] = temp[1]

        df_list.append(temp[["content", "label"]])

    liar_df = pd.concat(df_list)

    return liar_df


# -----------------------------
# 3. CLEAN + STANDARDIZE
# -----------------------------
def clean_data(df):
    df = df.dropna()
    df["content"] = df["content"].astype(str)

    df = df.drop_duplicates(subset=["content"])

    df["content"] = df["content"].str.replace("\n", " ")
    df["content"] = df["content"].str.strip()

    return df


# -----------------------------
# 4. MERGE EVERYTHING
# -----------------------------
def merge_all():
    isot = download_kaggle()
    liar = download_hf()

    print("Merging datasets...")

    df = pd.concat([isot, liar])
    df = clean_data(df)

    output_path = os.path.join(DATA_DIR, "final_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Final dataset saved at: {output_path}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    merge_all()
# for data manipulation
import os
import pandas as pd
# for splitting the data
from sklearn.model_selection import train_test_split
# for reading from / writing to the Hugging Face dataset space
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO_ID = "Marco8000/visit-with-us-dataset"
DATA_DIR = "tourism_project/data"
TARGET = "ProdTaken"

# ---------------------------------------------------------------- 1. Load the dataset from the Hugging Face data space
raw_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="tourism.csv", repo_type="dataset", token=HF_TOKEN)
df = pd.read_csv(raw_path)
print("Loaded dataset from Hugging Face:", df.shape)

# ---------------------------------------------------------------- 2. Data cleaning
# Drop columns that carry no predictive information: the exported row index and the customer identifier
unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
df = df.drop(columns=unnamed_cols + ["CustomerID"])
print("Dropped columns:", unnamed_cols + ["CustomerID"])

# Fix the misspelled gender category
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Remove exact duplicate rows so the same customer profile cannot leak from train into test
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_before - len(df)} duplicate rows")

# Fill missing values (none in the current file, but new data may contain gaps)
numeric_cols = df.select_dtypes(include="number").columns.drop(TARGET)
categorical_cols = df.select_dtypes(exclude="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values after cleaning:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------- 3. Train / test split (stratified on the target)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])

os.makedirs(DATA_DIR, exist_ok=True)
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print("Purchase rate - train: {:.3f}, test: {:.3f}".format(train_df[TARGET].mean(), test_df[TARGET].mean()))

# ---------------------------------------------------------------- 4. Upload the splits to the Hugging Face data space
api = HfApi(token=HF_TOKEN)
for local_path, name in [(train_path, "train.csv"), (test_path, "test.csv")]:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=name,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload {name}",
    )
print(f"Uploaded train.csv and test.csv to https://huggingface.co/datasets/{DATASET_REPO_ID}")

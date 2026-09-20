# for data manipulation
import os
import pandas as pd
# for registering the dataset on the Hugging Face Hub
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO_ID = "Marco8000/visit-with-us-dataset"
RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset validated successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("ProdTaken distribution:")
print(df["ProdTaken"].value_counts())

# Create the dataset repository on the Hugging Face Hub (no-op if it already exists)
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False, exist_ok=True, token=HF_TOKEN)

# Upload the raw dataset
api.upload_file(
    path_or_fileobj=RAW_PATH,
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    commit_message="Register raw tourism dataset",
)
print(f"Dataset registered at https://huggingface.co/datasets/{DATASET_REPO_ID}")

import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = "Marco8000/visit-with-us-app"
DEPLOYMENT_DIR = "tourism_project/deployment"

api = HfApi(token=HF_TOKEN)

# ---------------------------------------------------------------- 1. Create the Docker Space if it does not exist yet
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
    print(f"Space {SPACE_REPO_ID} already exists.")
except RepositoryNotFoundError:
    try:
        create_repo(repo_id=SPACE_REPO_ID, repo_type="space", space_sdk="docker", private=False, token=HF_TOKEN)
        print(f"Created Docker Space {SPACE_REPO_ID}")
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code == 402:
            raise RuntimeError(
                "Hugging Face requires a PRO subscription to create Docker Spaces on cpu-basic. "
                "Upgrade the account, then rerun this script to create the required Streamlit Docker Space."
            ) from error
        raise

# ---------------------------------------------------------------- 2. Write the Space README with the required Docker metadata
readme = """---
title: Visit with Us Wellness Package Predictor
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

Streamlit app that predicts whether a customer will purchase the Wellness Tourism Package.
The model is downloaded from https://huggingface.co/Marco8000/visit-with-us-model at start-up.
"""
with open(os.path.join(DEPLOYMENT_DIR, "README.md"), "w", encoding="utf-8") as file:
    file.write(readme)

# ---------------------------------------------------------------- 3. Push all deployment files to the Space
api.upload_folder(
    folder_path=DEPLOYMENT_DIR,
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    commit_message="Deploy Streamlit app",
)
print(f"Deployment files pushed to https://huggingface.co/spaces/{SPACE_REPO_ID}")

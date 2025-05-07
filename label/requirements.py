import os
import git
import subprocess
from huggingface_hub import hf_hub_download

REPO_URL = "https://github.com/facebookresearch/watermark-anything.git"
REPO_BRANCH = 'bbec07bca82a416e5a6ff9d75a295cae5c166aaf'
LOCAL_PATH = "./watermark-anything"
MODEL_ID = "facebook/watermark-anything"

def install_src():
    if not os.path.exists(LOCAL_PATH):
        print(f"Cloning repository from {REPO_URL}...")
        repo = git.Repo.clone_from(REPO_URL, LOCAL_PATH)
        repo.git.checkout(REPO_BRANCH)
    else:
        print(f"Repository already exists at {LOCAL_PATH}")

    requirements_path = os.path.join(LOCAL_PATH, "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing requirements...")
        subprocess.check_call(["pip", "install", "-r", requirements_path])
    else:
        print("No requirements.txt found.")

def install_model():
    checkpoint_path = os.path.join(LOCAL_PATH, "checkpoints")
    hf_hub_download(repo_id=MODEL_ID, filename='checkpoint.pth', local_dir=checkpoint_path)

# clone repo and download model
install_src()
install_model()

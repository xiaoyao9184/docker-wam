import os
import sys
import git
import subprocess
from huggingface_hub import hf_hub_download

REPO_URL = "https://github.com/facebookresearch/watermark-anything.git"
REPO_BRANCH = '45d56c2b61f2bc73caeafc90e14df33ad50b238c'
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

# fix sys.path for import
print(f"LOCAL_PATH: {os.path.abspath(LOCAL_PATH)}")
os.environ["APP_PATH"] = os.path.abspath(LOCAL_PATH)

# run gradio
import re
import sys
from gradio.cli import cli
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.argv.append(re.sub(r'app\.py$', 'gradio_app.py', sys.argv[0]))
    sys.exit(cli())

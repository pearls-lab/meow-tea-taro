from huggingface_hub import HfApi, create_repo
HF_REPO_ID = "ruiyiwang/SFT-alfworld-text-visual-Qwen2.5-VL-7B-Instruct"
LOCAL_DIR = "/root/meow-tea-taro/examples/alfworld/checkpoints/"

create_repo(HF_REPO_ID, exist_ok=True, repo_type="model")

api = HfApi()
api.upload_large_folder(
    folder_path=LOCAL_DIR,
    repo_id=HF_REPO_ID,
    repo_type="model",
)
print("Upload completed successfully!")
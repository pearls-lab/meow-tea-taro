# Copyright 2025 Ruiyi Wang, PEARLS Lab, UC San Diego
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from omegaconf import DictConfig


def upload_val_results_to_hf(config: DictConfig):
    """
    Upload/Update validation results from trainer.validation_data_dir to HF repo."""
    from huggingface_hub import create_repo, HfApi
    try:
        # Create HF repo if not exists
        api = HfApi()
        create_repo(
            config.trainer.hf_kwargs.save_hf_repo_id, 
            repo_type="model",
            exist_ok=True
        )
        # Upload folder
        api.upload_folder(
            folder_path=config.trainer.validation_data_dir,
            repo_id=config.trainer.hf_kwargs.save_hf_repo_id,
            repo_type="model",
            path_in_repo="val_results"
        )
        print(f"Uploaded val results {config.trainer.validation_data_dir} to HF repo {config.trainer.hf_kwargs.save_hf_repo_id}.")
    except:
        print("Cannot upload val results to HF.")


def upload_ckpt_to_hf(config: DictConfig):
    """
    Upload/Update checkpoints from trainer.default_local_dir to HF repo.
    """
    from huggingface_hub import create_repo, HfApi
    import os
    try:
        # Create HF repo if not exists
        api = HfApi()
        create_repo(
            config.trainer.hf_kwargs.save_hf_repo_id, 
            repo_type="model",
            exist_ok=True
        )
        # Upload the entire local checkpoint folder
        api.upload_large_folder(
            folder_path=config.trainer.default_local_dir,
            repo_id=config.trainer.hf_kwargs.save_hf_repo_id,
            repo_type="model",
        )
        print(f"Uploaded folder {config.trainer.default_local_dir} to HF repo {config.trainer.hf_kwargs.save_hf_repo_id}.")

        # Upload the local directory
        if os.path.exists("local"):
            api.upload_large_folder(
                folder_path="local",
                repo_id=config.trainer.hf_kwargs.save_hf_repo_id,
                repo_type="model",
                path_in_repo="local"
            )
            print(f"Uploaded folder local to HF repo {config.trainer.hf_kwargs.save_hf_repo_id}.")
    except:
        print("Cannot upload checkpoints to HF.")


def download_ckpt_from_hf(config: DictConfig):
    """
    Download the latest checkpoint (name stored in a txt file) from HF repo.
    Should be the same HF repo you upload to during training.
    """
    from huggingface_hub import snapshot_download, hf_hub_download
    import os
    try:
        # First, try downloading latest checkpointed iteration from HF
        latest_ckpt_filename = "latest_checkpointed_iteration.txt"
        hf_hub_download(
            repo_id=config.trainer.hf_kwargs.save_hf_repo_id,
            repo_type="model",
            filename=latest_ckpt_filename,
            local_dir=config.trainer.default_local_dir,
        )
        # If the latest checkpointed iteration file exists, download the latest checkpoint to local checkpoints folder only.
        ckpt_iter = open(os.path.join(config.trainer.default_local_dir, latest_ckpt_filename)).readline().strip()
        ckpt_name = f"global_step_{ckpt_iter}"
        snapshot_download(
            repo_id=config.trainer.hf_kwargs.save_hf_repo_id,
            repo_type="model",
            local_dir=config.trainer.default_local_dir,
            allow_patterns=f"{ckpt_name}/*"
        )
        # self.global_steps = int(ckpt_iter)
        print(f"Downloaded folder {config.trainer.default_local_dir}/{ckpt_name} from {config.trainer.hf_kwargs.save_hf_repo_id}.")
    except:
        print("Previous checkpoints not found on HF.")


def resume_wandb_logs(config: DictConfig):
    """
    Download wandb run logs from HF repo to trainer.default_local_dir.
    Update the wandb config before initializing the logger.
    """
    import json
    from omegaconf import open_dict
    from huggingface_hub import hf_hub_download
    import os
    # First, try downloading the wandb_run_info.json file from HF repo
    try:
        wandb_run_info_filename = "wandb_run_info.json"
        hf_hub_download(
            repo_id=config.trainer.hf_kwargs.save_hf_repo_id,
            repo_type="model",
            filename=wandb_run_info_filename,
            local_dir=config.trainer.default_local_dir,
        )
        # If the wandb run info file exists, update the logger wandb config info.
        if "wandb" in config.trainer.logger:
            with open_dict(config.trainer):
                config.trainer.wandb_run_info = None
                wandb_info_file = os.path.join(config.trainer.default_local_dir,  wandb_run_info_filename)
                if os.path.exists(wandb_info_file):
                    try:
                        with open(wandb_info_file, 'r') as f:
                            config.trainer.wandb_run_info = json.load(f)
                    except:
                        print("Cannot obtain wandb run info.")
    except:
        print("Cannot find wandb run info file in HF.")


def save_wandb_logs(logger, config: DictConfig):
    """
    Save wandb run logs to trainer.default_local_dir
    """
    import os
    import json
    if "wandb" in config.trainer.logger:
        try:
            run_info = {
                "id": logger.logger["wandb"].run.id,
                "name": logger.logger["wandb"].run.name,
            }
            wandb_info_file = os.path.join(config.trainer.default_local_dir, "wandb_run_info.json")
            with open(wandb_info_file, 'w') as f:
                json.dump(run_info, f, indent=2)
            print("Saved wandb run logs.")
        except:
            print("Cannot save wandb run info.")



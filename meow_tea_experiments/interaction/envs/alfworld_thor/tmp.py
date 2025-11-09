# from pathlib import Path
# import shutil

# SPLIT = "valid_seen"
# SOURCE_ROOT = Path(f"/root/.cache/alfworld/json_2.1.1/{SPLIT}")
# DEST_PDDL = Path(f"/root/data/alfworld_tw/{SPLIT}")
# DEST_PDDL.mkdir(parents=True, exist_ok=True)

# traj_paths = sorted(SOURCE_ROOT.rglob("traj_data.json"))
# for idx, traj_path in enumerate(traj_paths, start=1):
#     pddl_path = traj_path.with_name("game.tw-pddl")
#     if not pddl_path.exists():
#         print((f"Missing {pddl_path} for index {idx}"))
#         continue
#     dest_path = DEST_PDDL / f"{SPLIT}_{idx}.tw-pddl"
#     shutil.copy2(pddl_path, dest_path)
#     print(f"{idx:04d}: {pddl_path} -> {dest_path}")

# print(f"Copied {len(traj_paths)} files into {DEST_PDDL}")


import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# --- Configuration ---

# 1. Path to your generated Parquet file
PARQUET_FILE_PATH = "/root/meow-tea-taro/examples/alfworld/local/train_parquet/train_v5_high_500.parquet"

# 2. The Hugging Face model identifier you are using for training.
#    This is crucial for loading the correct tokenizer.
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-VL-7B-Instruct" # Example: Use the actual model you're fine-tuning

# 3. The number of tokens a single image adds to the context.
#    - For LLaVA-1.5 (336px image, 14px patch): (336/14)^2 + 1 = 577
#    - Adjust this value based on your model's vision encoder configuration.
TOKENS_PER_IMAGE = 512

def calculate_token_lengths(parquet_path, model_path, tokens_per_image):
    """
    Calculates the total token length (text + image) for each sample in a Parquet file.

    Returns:
        A list of total token lengths for each sample.
    """
    print(f"Loading tokenizer for '{model_path}'...")
    # trust_remote_code is often needed for multimodal models
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Reading Parquet file from '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)

    all_token_lengths = []

    print("Calculating token lengths for each sample...")
    # Iterate through each row of the DataFrame with a progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        messages = row['messages']
        images = row['images']

        # --- Calculate Text Token Length ---
        # apply_chat_template correctly formats the conversation for tokenization
        # We don't add a generation prompt because we want the length of the full SFT example
        text_tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True
        )
        text_token_count = len(text_tokens)

        # --- Calculate Image Token Length ---
        num_images = len(images)
        image_token_count = num_images * tokens_per_image

        # --- Calculate Total Token Length ---
        total_length = text_token_count + image_token_count
        all_token_lengths.append(total_length)

    return all_token_lengths

if __name__ == "__main__":
    # Calculate the lengths for all samples
    token_lengths = calculate_token_lengths(PARQUET_FILE_PATH, MODEL_NAME_OR_PATH, TOKENS_PER_IMAGE)

    if not token_lengths:
        print("No samples found or processed.")
    else:
        # Use numpy for easy statistics
        lengths_np = np.array(token_lengths)

        max_len = np.max(lengths_np)
        avg_len = np.mean(lengths_np)
        median_len = np.median(lengths_np)
        p90_len = np.percentile(lengths_np, 90)
        p99_len = np.percentile(lengths_np, 99)

        print("\n--- Token Length Statistics ---")
        print(f"Total Samples Processed: {len(lengths_np)}")
        print(f"Tokens per Image (Assumed): {TOKENS_PER_IMAGE}")
        print("-" * 30)
        print(f"Max Token Length:    {int(max_len)}")
        print(f"Average Token Length:  {avg_len:.2f}")
        print(f"Median Token Length:   {int(median_len)}")
        print(f"90th Percentile:     {int(p90_len)}")
        print(f"99th Percentile:     {int(p99_len)}")
        print("---------------------------------")

        # This max length is a good starting point for setting `max_length` in your training config
        print(f"\nRecommendation: A `max_length` of at least {int(max_len)} would cover all samples.")
        print(f"A more practical `max_length` might be around the 99th percentile: {int(p99_len)}")
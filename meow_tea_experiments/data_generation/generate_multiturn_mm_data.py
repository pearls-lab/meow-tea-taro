from pyexpat.errors import messages
import pandas as pd
import json
from PIL import Image
import io
import os
from tqdm import tqdm

# Helper function to convert an image file to bytes
def get_image_bytes(image_path):
    """Opens an image, converts it to RGB, and returns its byte representation."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure consistent color format
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG') # You can use other formats like JPEG
        return byte_arr.getvalue()


def format_input_for_sft(traj_file_path: str, image_file_path: str) -> None:
    """Formats a single training sample for chat-based SFT with images.

    Args:
        traj_file_path (str): Path to the trajectory text file.
        image_file_path (str): Path to the associated image file.
        turn (int): The turn number in the conversation (0-indexed).
    """
    with open(traj_file_path, 'r') as f:
        traj = json.load(f)

    data = []
    messages = []
    images = []
    for i in range(len(traj)//2):
        user_format = "\ncurrent state: <image>\n\nyour action: "
        messages.append(
            {"role": "user", "content": user_format})
        messages.append(
            {"role": "assistant", "content": traj[i*2 + 1]})
        curr_img = get_image_bytes(os.path.join(image_file_path, f"step_{i}.png"))
        images.append(curr_img)
        data.append(
            {
                "messages": messages.copy(),
                "images": images.copy()
            }
        )

    return data

SPLIT = "train_v5"
TRAJ_DIR = f"/root/data/alfworld/{SPLIT}_high_trajs"
IMAGE_DIR = f"/root/data/alfworld/{SPLIT}_high_images/"
OUTPUT_PARQUET = f"/root/data/alfworld/{SPLIT}_high_visual.parquet"

file_list = sorted(os.listdir(TRAJ_DIR))
print(len(file_list))

for i in tqdm(range(500)):
    traj_file = file_list[i]
    traj_path = os.path.join(TRAJ_DIR, traj_file)
    image_path = os.path.join(IMAGE_DIR, os.path.splitext(traj_file)[0])
    data = format_input_for_sft(traj_path, image_path)
    df = pd.DataFrame(data)
    if not os.path.exists(OUTPUT_PARQUET):
        df.to_parquet(OUTPUT_PARQUET, index=False)
    else:
        existing_df = pd.read_parquet(OUTPUT_PARQUET)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(OUTPUT_PARQUET, index=False)

# --- Define your image paths ---
# Create dummy image files if you don't have them


# image1_path = "image1.png"
# image2_path = "image2.png"

# --- Data for the first training sample (Row 1) ---
# A two-turn conversation with one image
# messages_1 = [
#     {"role": "user", "content": "What do you see in this picture? <image>"},
#     {"role": "assistant", "content": "I see a red rectangle."}
# ]
# images_1 = [get_image_bytes(image1_path)]

# --- Data for the second training sample (Row 2) ---
# A four-turn conversation using two images.
# The order of <image> placeholders must match the order in the images list.
# messages_2 = [
#     {"role": "user", "content": "What do you see in this picture? <image>"},
#     {"role": "assistant", "content": "I see a red rectangle."},
#     {"role": "user", "content": "Great. Now what about this second one? <image>"},
#     {"role": "assistant", "content": "This one is a blue rectangle."}
# ]
# images_2 = [
#     get_image_bytes(image1_path),
#     get_image_bytes(image2_path)
# ]

# --- Create the DataFrame ---
# Each dictionary in this list corresponds to one row in the final Parquet file.
# data = [
#     {"messages": messages_1, "images": images_1},
#     {"messages": messages_2, "images": images_2},
# ]

# df = pd.DataFrame(data)

# # --- Save to a Parquet file ---
# df.to_parquet("sft_training_data.parquet", index=False)

# print("Parquet file 'sft_training_data.parquet' created successfully.")
# print("\nDataFrame structure:")
# print(df)
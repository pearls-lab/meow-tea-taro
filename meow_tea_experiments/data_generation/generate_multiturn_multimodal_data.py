from pyexpat.errors import messages
import pandas as pd
import json
from PIL import Image
import io
import os
from tqdm import tqdm
import random
import pickle  # Essential for handling lists of bytes in Parquet

# Generic format constants
TEXT_ONLY_USER_FORMAT = "current state: {text_obs}\n\nyour action: "
IMAGE_ONLY_USER_FORMAT = "current state: state shown in image: <image>\n\nyour action: "
TEXT_IMAGE_USER_FORMAT = "current state: {text_obs}\nstate shown in image: <image>\n\nyour action: "

def image_to_bytes(image):
    """Helper to convert PIL Image to bytes for Parquet storage."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def format_input_for_sft(traj_file_path: str, use_text=True, use_image=True) -> list:
    with open(traj_file_path, 'r') as f:
        traj = json.load(f)["trajectory"]

    data_rows = []
    
    # These lists accumulate history as we iterate through the trajectory
    history_messages = []
    history_images = [] 

    for i in range(len(traj)):
        item = traj[i]
        
        # --- 1. Handle Observations (User) ---
        if item["name"] == "state_text":
            if use_text and use_image:
                history_messages.append({
                    "role": "user", 
                    "content": TEXT_IMAGE_USER_FORMAT.format(text_obs=item["content"])
                })
            elif use_text:
                history_messages.append({
                    "role": "user", 
                    "content": TEXT_ONLY_USER_FORMAT.format(text_obs=item["content"])
                })
            elif use_image:
                if i == 0:
                    history_messages.append({
                        "role": "user", 
                        "content": TEXT_IMAGE_USER_FORMAT.format(text_obs=item["content"])
                    })
                else:
                    history_messages.append({
                        "role": "user", 
                        "content": IMAGE_ONLY_USER_FORMAT
                    })
            
        elif item["name"] == "state_image":
            if use_image:
                image_file_path = item["content"]
                if os.path.exists(image_file_path):
                    with Image.open(image_file_path).convert("RGB") as img:
                        history_images.append(image_to_bytes(img))
                else:
                    # Dummy black image if missing to prevent index misalignment
                    dummy = Image.new('RGB', (100, 100), (0, 0, 0))
                    history_images.append(image_to_bytes(dummy))

        # --- 2. Handle Actions (Assistant) -> GENERATE ROW ---
        elif item["name"] == "action":
            # Create a copy of the current messages so far
            current_step_messages = history_messages.copy()
            
            # Append the target action for this step
            current_step_messages.append({
                "role": "assistant", 
                "content": item["content"]
            })

            # --- SERIALIZATION (Prevent PyArrow Crash) ---
            # We dump to string/bytes immediately before adding to the row
            messages_str = json.dumps(current_step_messages)
            images_blob = pickle.dumps(history_images) if use_image else None
            
            row = {"messages": messages_str}
            if use_image:
                row["images"] = images_blob
            
            data_rows.append(row)

            # IMPORTANT: Update the history for the NEXT step
            # The action we just learned becomes history for the next turn
            history_messages.append({
                "role": "assistant", 
                "content": item["content"]
            })

    return data_rows

# --- Main Execution ---
SPLIT = "train"
TRAJ_DIR = f"/root/data/alfworld/{SPLIT}"
OUTPUT_PARQUET = f"/root/data/alfworld/{SPLIT}_visual.parquet"

# Load pass list or fallback to directory scan
try:
    with open("/root/data/alfworld/sft_info/pass_list_sorted.json", "r") as f:
        pass_list = json.load(f)
except:
    print("Pass list not found, scanning directory...")
    pass_list = [f.replace('.traj.json', '') for f in os.listdir(TRAJ_DIR) if f.endswith('.traj.json')]

all_data = []
print("Processing trajectories step-by-step...")
for i in tqdm(range(len(pass_list))):
    traj_file = pass_list[i]
    traj_path = os.path.join(TRAJ_DIR, f"{traj_file}.traj.json")
    
    if os.path.exists(traj_path):
        data = format_input_for_sft(traj_path, use_text=False, use_image=True)
        all_data.extend(data)

random.shuffle(all_data)
df = pd.DataFrame(all_data)
print(all_data[0]["messages"])

if not os.path.exists(OUTPUT_PARQUET):
    print("Saving to Parquet (using pyarrow engine)...")
    # Using 'pyarrow' engine is cleaner for binary data
    df.to_parquet(OUTPUT_PARQUET, index=False, engine='pyarrow')

print(f"Success! Saved {len(df)} rows.")
print("Sample row keys:", df.iloc[0].keys())
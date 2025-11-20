# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from typing import Union, Optional, List, Dict, Any

import pandas as pd
import re
import torch
import json
import pickle
import io
from PIL import Image
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


# --- HELPER: Robust Image Processor ---
def process_image(image: Union[dict, Image.Image, bytes]) -> Image.Image:
    """
    Converts various input formats (Bytes, Dict, PIL) into a standard RGB PIL Image.
    Essential for handling the Pickled bytes from the Parquet file.
    """
    # Case 1: Already a PIL Image
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # Case 2: Raw Bytes (From our Pickled Parquet column)
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")

    # Case 3: Dictionary Wrapper (Legacy verl format)
    if isinstance(image, dict):
        if "bytes" in image:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if "path" in image:
            return Image.open(image["path"]).convert("RGB")

    raise ValueError(f"Unsupported image format: {type(image)}")


class ChatBasedSFTDataset(Dataset):
    """
    This is an in-memory SFTDataset that supports:
    1. Standard Text SFT
    2. Multimodal SFT (via Generic <image> tags)
    3. "Flattened" Parquet files (JSON strings + Pickled Bytes) to avoid PyArrow errors
    """

    def __init__(
        self, 
        parquet_files: Union[str, ListConfig],
        tokenizer,
        config,
        processor: Optional[ProcessorMixin] = None,
    ):
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        self.use_shm = config.get("use_shm", False)

        chat_based_config = config.get("chat_based", {})
        self.messages_key = chat_based_config.get("messages_key", "messages")
        self.image_key = chat_based_config.get("image_key", "images")
        self.video_key = chat_based_config.get("video_key", "videos")
        self.apply_chat_template_kwargs = chat_based_config.get("apply_chat_template_kwargs", {})
        self.return_multi_modal_inputs = chat_based_config.get("return_multi_modal_inputs", True)

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_process(self):
        """
        Reads the Parquet files. 
        Note: We use standard pd.read_parquet here. Since we serialized the complex columns 
        (Messages -> JSON String, Images -> Pickled Bytes) in the generator, 
        this will no longer crash with PyArrow chunk errors.
        """
        def series_to_item(ls):
            if isinstance(ls, list):
                return ls
            return ls[0]

        dataframes = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)
        self.dataframe = pd.concat(dataframes)

        # Store records as a list of dicts for easy access
        self.dataframe_dict = self.dataframe.to_dict(orient="records")
        # Note: We don't pre-process self.messages here anymore because they need dynamic deserialization

    def _truncate_system_template(self, text: str) -> str:
        pattern = r'<\|im_start\|>system\n.*?<\|im_end\|>\n'
        return re.sub(pattern, '', text, flags=re.DOTALL)

    def _build_messages(self, example: dict) -> List[Dict]:
        """
        Extracts messages and handles:
        1. JSON Deserialization (String -> List)
        2. Expanding generic <image> tags into structured content dicts
        """
        raw_content = example.get(self.messages_key)
        
        # --- FIX 1: Deserialize JSON String if necessary ---
        if isinstance(raw_content, str):
            try:
                messages = json.loads(raw_content)
            except json.JSONDecodeError:
                # Fallback for pure text rows
                messages = [{"role": "user", "content": raw_content}]
        else:
            messages = raw_content

        # --- FIX 2: Expand <image> placeholders ---
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                if not isinstance(content, str):
                    continue
                
                content_list = []
                # Split text by <image> or <video> tags
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages
    
    def __len__(self):
        return len(self.dataframe_dict)

    def __getitem__(self, item):
        row_dict = self.dataframe_dict[item].copy()
        
        # 1. Build structured messages (handles JSON parsing)
        messages = self._build_messages(row_dict)
        
        model_inputs = {}

        if self.processor:
            raw_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
            )
            
            multi_modal_data = {}
            processed_images = None
            
            # --- FIX 3: Handle Image Deserialization (Pickle -> List -> PIL) ---
            row_dict_images = row_dict.pop(self.image_key, None)
            
            if row_dict_images is not None:
                images_list = []
                
                # If it's bytes, unpickle it (This is the "Nuclear" fix)
                if isinstance(row_dict_images, bytes):
                    try:
                        images_list = pickle.loads(row_dict_images)
                    except Exception as e:
                        print(f"Error unpickling images: {e}")
                        images_list = []
                elif isinstance(row_dict_images, list):
                    images_list = row_dict_images
                
                # Convert Bytes/Dicts to proper PIL images for the Processor
                if images_list:
                    processed_images = [process_image(image) for image in images_list]
                    multi_modal_data["image"] = processed_images

            # Handle Videos (Assuming standard list format, but easy to adapt if pickled too)
            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                # Import here to avoid circular dependency if process_video is in utils
                from verl.utils.dataset.vision_utils import process_video
                videos = [process_video(video) for video in row_dict_videos]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            # Call the Hugging Face Processor
            model_inputs = self.processor(
                text=[raw_text], 
                images=processed_images, 
                videos=videos, 
                return_tensors="pt"
            )
            
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            # Text-only fallback
            raw_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_text, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # --- Prompt Masking Logic ---
        prompt_chat = messages[:-1]
        if self.processor:
            prompt_chat_str = self.processor.apply_chat_template(
                prompt_chat, add_generation_prompt=True, tokenize=False
            )
            prompt_model_inputs = self.processor(text=prompt_chat_str, add_special_tokens=False, return_tensors="pt")
            prompt_ids = prompt_model_inputs["input_ids"][0]
        else:
            prompt_chat_str = self.tokenizer.apply_chat_template(
                prompt_chat, add_generation_prompt=True, tokenize=False
            )
            prompt_ids = self.tokenizer.encode(prompt_chat_str, add_special_tokens=False)
        
        prompt_length = len(prompt_ids)

        # Post-process (padding, truncation)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            truncation=self.truncation,
        )
        input_ids = input_ids[0]
        attention_mask = attention_mask[0]

        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0))[0]

        loss_mask = attention_mask.clone()
        if prompt_length > 0:
            loss_mask[:prompt_length] = 0

        loss_mask[input_ids == self.tokenizer.pad_token_id] = 0

        if attention_mask.sum() > 0:
            last_token_idx = attention_mask.sum() - 1
            loss_mask[last_token_idx] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
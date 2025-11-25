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


from collections import defaultdict
import math

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register
from verl.experimental.reward.reward_loop.registry import register as register_experimental
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


@register("agentic_heuristics")
class AgenticHeuristicsRewardManager:
    """The reward manager design for multi-turn agentic tasks with verified feedback."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        print("Initialized AgenticHeuristicsRewardManager")
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        print(f"DEBUG: AgenticHeuristicsRewardManager called. Batch size: {len(data)}")

        # NOTE from meow-tea: We intentionally do NOT check for rm_scores here because for agentic environments,
        # the reward may be pre-computed during the agent loop and stored in rm_scores.
        # However, we want to use the final_rewards and interm_rewards from the non_tensor_batch
        # to properly assign rewards based on the heuristic (e.g., dense intermediate rewards).
        # If you want to use the pre-computed rm_scores instead, uncomment the block below:
        #
        # if "rm_scores" in data.batch.keys():
        #     print("DEBUG: rm_scores found in batch. Returning existing scores.")
        #     if return_dict:
        #         return {"reward_tensor": data.batch["rm_scores"]}
        #     else:
        #         return data.batch["rm_scores"]

        # print("DEBUG: Computing rewards from final_rewards.")
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        response_length = reward_tensor.shape[1]  # Get the response tensor length

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # we already compute final/intermediate rewards during multiturn rollout
            final_score = data_item.non_tensor_batch["final_rewards"]
            
            # Convert to int for indexing
            valid_response_length_int = int(valid_response_length)
            
            # Convert final_score to float and check for NaN
            final_score_float = float(final_score)
            if math.isnan(final_score_float):
                with open("debug.log", "a") as f:
                    f.write(f"WARNING: Item {i} has NaN final_score. Setting to 0.0.\n")
                final_score_float = 0.0
            
            # Debug logging
            with open("debug.log", "a") as f:
                f.write(f"Item {i}: final_score={final_score_float}, valid_response_length={valid_response_length_int}, response_length={response_length}\n")
            
            # Bounds check and assign reward at the last valid token position
            if valid_response_length_int > 0 and valid_response_length_int <= response_length:
                reward_idx = valid_response_length_int - 1
                reward_tensor[i, reward_idx] = final_score_float
            else:
                with open("debug.log", "a") as f:
                    f.write(f"WARNING: Item {i} has invalid valid_response_length={valid_response_length_int}, response_length={response_length}. Skipping reward assignment.\n")
            
            # data_source = data_item.non_tensor_batch[self.reward_fn_key]
            data_source = "sweagent_tasks"
            # ground_truth = data_item.non_tensor_batch["extra_info"]["response"]
            ground_truth  = data_item.non_tensor_batch.get("uid", "unknown_uid")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(final_score, dict):
                    for key, value in final_score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", final_score)


        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@register_experimental("agentic_heuristics")
class AgenticHeuristicsRewardLoopManager(RewardLoopManagerBase):
    """The reward loop manager for agentic heuristics tasks."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)

    async def run_single(self, data: DataProto) -> dict:
        # Return None to defer reward computation to RayPPOTrainer
        return {"reward_score": None, "reward_extra_info": {}}

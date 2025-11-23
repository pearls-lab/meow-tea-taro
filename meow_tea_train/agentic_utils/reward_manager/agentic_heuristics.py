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

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register
from verl.experimental.reward.reward_loop import register as register_experimental
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

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

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
            interm_scores = data_item.non_tensor_batch["interm_rewards"]
            sep_token_positions = data_item.non_tensor_batch["sep_token_positions"]
            
            response_length = data_item.batch["prompts"].shape[-1]
            
            if len(sep_token_positions) > 0: # edge case:
                if data_item.non_tensor_batch["extra_info"]["reward_method"] == "dense":
                    # Assign dense reward to each sep_pos of the sequence
                    # Scenario 1: agent wins the game, average intermediate reward against total num of non-zero rewards (sum up to 1)
                    if final_score > 0:
                        num_rewards = len([x for x in interm_scores if x > 0])
                    # Scenario 2: agent loses the game, average intermediate reward against pre-defined max num of rewards (sum up to 1)
                    else:
                        num_rewards = data_item.non_tensor_batch["max_total_rewards"]
                    num_rewards = max(1, num_rewards) # edge case where agent wins but receives no immediate rewards
                    for k in range(len(sep_token_positions)):
                        if sep_token_positions[k] < response_length: # edge case
                            reward_tensor[i, sep_token_positions[k]] = interm_scores[k] / num_rewards
                else:
                    # Assign sparse reward to the last sep_pos of the sequence
                    if sep_token_positions[-1] < response_length: # edge case
                        reward_tensor[i, sep_token_positions[-1]] = final_score
                    
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            ground_truth = data_item.non_tensor_batch["extra_info"]["response"]

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
        print("Initialized AgenticHeuristicsRewardLoopManager")
        self.num_examine = 1
        self.reward_fn_key = config.data.get("reward_fn_key", "data_source")
        self.already_print_data_sources = {}

    async def run_single(self, data: DataProto) -> dict:
        data_item = data[0]

        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        final_score = data_item.non_tensor_batch["final_rewards"]

        data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
        ground_truth = data_item.non_tensor_batch.get("extra_info", {}).get("response", "unknown")

        if data_source not in self.already_print_data_sources:
            self.already_print_data_sources[data_source] = 0

        if self.already_print_data_sources[data_source] < self.num_examine:
            self.already_print_data_sources[data_source] += 1
            print("[prompt]", prompt_str)
            print("[response]", response_str)
            print("[ground_truth]", ground_truth)
            if isinstance(final_score, dict):
                for key, value in final_score.items():
                    print(f"[{key}]", value)
            else:
                print("[score]", final_score)

        # Ensure final_score is a float for the return value
        reward_score = final_score
        if isinstance(final_score, dict):
             # If it's a dict, we can't return it as the single scalar score.
             # Assuming there's a 'score' key or similar, or we just return 0.0 and rely on extra_info
             reward_score = final_score.get('score', 0.0)

        return {"reward_score": reward_score, "reward_extra_info": {}}

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


import random
import os
import json
import argparse
from typing import List, Dict

from meow_tea_experiments.data_generation.multiturn_template import MULTITURN_PROMPT_TEMPLATE, SEP_TOKEN
from meow_tea_experiments.data_generation.utils.env_wrapper import EnvTrajectoryCollector


def format_multiturn_prompt(input_traj: List[str], objective: str) -> str:
    assert len(input_traj) >= 1
    assert len(input_traj) % 2 == 1

    interactions = ""

    for i in range(len(input_traj)):
        if i % 2 == 0:
            interactions += f"current state: {input_traj[i]}\n\n"
        else:
            interactions += f"your action: {input_traj[i]}"

    prompt = MULTITURN_PROMPT_TEMPLATE.format(task=objective, interactions=interactions)
    prompt += "your action: "

    return prompt


def format_multiturn_response(output_traj: List[str]) -> str:
    response = ""
    for i in range(len(output_traj)):
        response += output_traj[i] + SEP_TOKEN
    
    return response


def format_sft_messages(traj: List[str]) -> List[Dict[str, str]]:
    assert len(traj) % 2 == 1
    messages = []
    for i in range(len(traj)):
        if i % 2 == 1:
            messages.append({
                "role": "user",
                "content": f"current state: {traj[i]}\n\nyour action: "
            })
        else:
            messages.append({
                "role": "assistant",
                "content": traj[i]
            })
    
    return messages


def get_ppo_datapoint(trajectory: List[str], objective: str) -> Dict[str, str]:
    return {
        "prompt": format_multiturn_prompt(input_traj=trajectory[:1], objective=objective),
        "response": format_multiturn_response(output_traj=trajectory[1::2])
    }


def get_sft_datapoint(trajectory: List[str], objective: str) -> List[List[Dict[str, str]]]:
    """
    Returns a list of message lists, each building on the previous:
    - Item 1: [user, assistant]
    - Item 2: [user, assistant, user, assistant]
    - Item 3: [user, assistant, user, assistant, user, assistant]
    - etc.
    """
    all_messages = []
    current_messages = []
    
    for i in range(len(trajectory) // 2):
        # Add user message with current state
        current_messages.append({
            "role": "user",
            "content": format_multiturn_prompt(input_traj=trajectory[2*i:2*i+1], objective=objective)
        })
        # Add assistant response (the action)
        current_messages.append({
            "role": "assistant",
            "content": trajectory[2*i+1]
        })
        # Append a copy of the current conversation
        all_messages.append(list(current_messages))
    
    return all_messages


def generate_multiturn_data(env_name: str, instance_dir: str, instance_start: int, instance_end: int, task_prefix: str, train_type: str, **kwargs):
    datapoints = []

    for instance_id in range(instance_start, instance_end):
        collector = EnvTrajectoryCollector(env_name=env_name,
                                            instance_dir=instance_dir,
                                            instance_id=instance_id,
                                            task_prefix=task_prefix)
        gold_actions = collector.game_logistics["walkthrough"]
        objective = collector.game_logistics["objective"]
        trajectory = collector.get_trajectory_given_actions(gold_actions)

        if train_type == "ppo":
            ppo_datapoint = get_ppo_datapoint(trajectory=trajectory, objective=objective)
            ppo_datapoint["task_prefix"] = task_prefix
            ppo_datapoint["instance_id"] = instance_id
            datapoints.append(ppo_datapoint)
        elif train_type == "sft":
            sft_datapoint = get_sft_datapoint(trajectory=trajectory, objective=objective)
            datapoints += sft_datapoint
    
    return datapoints


def main(args):
    datapoints = generate_multiturn_data(env_name=args.env_name,
                                        instance_dir=args.instance_dir,
                                        instance_start=args.instance_id_range[0],
                                        instance_end=args.instance_id_range[1] + 1,
                                        task_prefix=args.task_prefix,
                                        train_type=args.train_type)
    if args.split == "train":
        random.shuffle(datapoints)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.train_type == "ppo":
        with open(os.path.join(args.out_dir, f"{args.split}.jsonl"), 'w') as f:
            for data in datapoints:
                f.write(json.dumps(data) + '\n')
    elif args.train_type == "sft":
        with open(os.path.join(args.out_dir, f"{args.split}.json"), 'w') as f:
            json.dump({"messages": datapoints}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-turn dataset for meow_tea_train.")
    parser.add_argument("--env_name", type=str, required=True, choices=["textworld", "alfworld"], help="The agentic environment.")
    parser.add_argument("--instance_dir", type=str, required=True, help="Path to the directory containing all instances.")
    parser.add_argument("--instance_id_range", type=int, nargs=2, required=True, metavar=("START", "END"), help="Instance ID range [start, end], inclusive.")
    parser.add_argument("--task_prefix", type=str, required=True, help="Task prefix of the instances, e.g. w2-o3-q4.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for the multi-turn dataset.")
    parser.add_argument("--train_type", type=str, required=True, choices=["sft", "ppo"], help="Types of training.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"], help="Data split: [train, validation, test]")
    
    args = parser.parse_args()
    main(args)
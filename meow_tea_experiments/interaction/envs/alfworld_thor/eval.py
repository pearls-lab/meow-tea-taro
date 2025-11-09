import json
import yaml
import os
from PIL import Image
import io
import base64
from tqdm import tqdm

from alfworld.env.thor_env_v5 import ThorEnv # type: ignore
from alfworld.agents.controller import OracleAgent # type: ignore
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# --- VLLM Model and Tokenizer Initialization ---
# TODO: Update with the path to your fine-tuned model
# ---


with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

DATA_DIR = f"/root/data/alfworld/valid_seen/"
DATA_TRJ_DIR = f"/root/data/alfworld/valid_seen_v5_high_trajs/"
MAX_TURNS = 10


# Helper function to convert an image to a base64 string
def get_image_base64(image: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    return base64.b64encode(byte_arr.getvalue()).decode('utf-8')

def run_inference(processor, llm, env: ThorEnv, traj_file: str) -> list:
    """
    Runs inference for a single trajectory.

    Args:
        env (ThorEnv): The ALFWorld environment.
        traj_data (dict): The trajectory data for the task.

    Returns:
        list: The generated trajectory of observations and actions.
    """
    traj_root = os.path.dirname(traj_file)
    with open(os.path.join(DATA_DIR, traj_file)) as f:
        traj_data = json.load(f)

    # Reset environment to the initial state of the task
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    scene_name = f'FloorPlan{scene_num}'

    env.reset(scene_name)
    env.restore_scene(traj_data, object_poses, object_toggles, dirty_and_empty)
    event = env.step(dict(traj_data['scene']['init_action']))

    class args: pass
    args.reward_config = 'rewards.json'
    env.set_task(traj_data, args)

    task_desc = traj_data['turk_annotations']['anns'][0]['task_desc']
    initial_obs = f"You are in a room. Your task is to: {task_desc}."
    print(f"Task: {task_desc}")

    controller_type = config['controller']['type']
    goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
    load_receps =   config['controller']['load_receps']
    debug = config['controller']['debug']

    controller = OracleAgent(env, traj_data, traj_root,
                            load_receps=load_receps, debug=debug,
                            goal_desc_human_anns_prob=goal_desc_human_anns_prob)
    
    initial_obs = controller.intro

    # Initialize conversation history
    messages = []
    images = []
    trajectory = [initial_obs]
    
    user_format = "\ncurrent state: <|vision_start|><|image_pad|><|vision_end|>\n\nyour action: "
    messages.append({"role": "user", "content": trajectory[0] + user_format})
    # messages.append({"role": "user", "content": user_format})
    curr_img = event.frame if isinstance(event.frame, Image.Image) else Image.fromarray(event.frame)
    images.append(get_image_base64(curr_img))

    # Limit the number of steps to prevent infinite loops
    for t in range(MAX_TURNS):
        # --- Prepare model input ---
        # Use the tokenizer's chat template to format the conversation history.
        # This ensures consistency with the training format.
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("--- Model Prompt ---")
        print(prompt)

        # VLLM expects image data in a separate dictionary
        image_data = {"image": images} if images else None
        print(len(image_data["image"]) if image_data else 0)

        # --- Get model's action ---
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50, stop=["\n", processor.tokenizer.eos_token])
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": image_data,
            },
            sampling_params=sampling_params
        )
        action_str = outputs[0].outputs[0].text.strip()
        print(f"Step {t}: Model Action: {action_str}")

        # --- Execute action in environment ---
        feedback = controller.step(action_str)
        event = env.last_event
        
        # Append action and feedback to trajectory
        trajectory.append(action_str)
        trajectory.append(feedback)

        # Update conversation history for the next turn
        messages.append({"role": "assistant", "content": action_str})

        # Check for goal satisfaction or failure
        if env.get_goal_satisfied():
            print("Goal satisfied!")
            break
        if not event.metadata['lastActionSuccess']:
            print(f"\tAction failed: {event.metadata['errorMessage']}")

        # --- Prepare for next turn ---
        # Get the current image frame
        curr_img = event.frame if isinstance(event.frame, Image.Image) else Image.fromarray(event.frame)
        images.append(get_image_base64(curr_img))

        # Format the user message for the next turn, consistent with SFT
        user_format = "\ncurrent state: <|vision_start|><|image_pad|><|vision_end|>\n\nyour action: "
        messages.append({"role": "user", "content": trajectory[t * 2] + user_format})
        # messages.append({"role": "user", "content": user_format})

    return trajectory


def main():
    SPLIT = "valid_seen_v5"
    OUT_DIR = f"/root/data/alfworld/{SPLIT}_vlm_inference_trajs_both/"
    os.makedirs(OUT_DIR, exist_ok=True)

    MODEL_PATH = "/root/checkpoints/global_step_795/"
    llm = LLM(model=MODEL_PATH, trust_remote_code=True, tensor_parallel_size=1)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Initialize environment once
    env = ThorEnv(scene="FloorPlan1")

    file_list = sorted(os.listdir(DATA_DIR))
    DATA_LEN = len(file_list) # Number of tasks to evaluate

    for i in tqdm(range(DATA_LEN)):
        traj_file = file_list[i]
        if not traj_file.endswith('.json'):
            continue

        if traj_file not in os.listdir(DATA_TRJ_DIR):
            print(f"Trajectory file not found for: {traj_file}")
            continue

        try:
            generated_trajectory = run_inference(processor, llm, env, traj_file)
            print(generated_trajectory)
            # Save the generated trajectory
            out_file = os.path.join(OUT_DIR, traj_file)
            with open(out_file, 'w') as out_f:
                json.dump(generated_trajectory, out_f, indent=4)

        except Exception as e:
            print(f"Exception occurred while processing {traj_file}: {e}")

    env.stop()
    print("Inference complete.")

if __name__ == "__main__":
    main()
import json
import yaml
import os
import sys
import multiprocessing as mp
from typing import List, Tuple
from PIL import Image
from pathlib import Path
import fcntl

# Add project root to Python path
# ROOT: meow_tea_gym/envs/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alfred.env.thor_env import ThorEnv
from alfworld.agents.controller import OracleAgent

class MockArgument:
    """Fake argument class to reuse existing functions"""
    reward_config = 'alfworld/rewards.json'


def safe_append_to_file(filepath: str, content: str):
    """
    Thread-safe append to file using file locking.

    Args:
        filepath: Path to the file
        content: Content to append (should include newline if needed)
    """
    with open(filepath, 'a') as f:
        # Acquire exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure it's written to disk
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def save_frame(event, img_dir, img_name):
    """Save the current frame and return the path"""
    if hasattr(event, 'frame') and event.frame is not None:
        frame_path = os.path.join(img_dir, f'{img_name}.png')
        if isinstance(event.frame, Image.Image):
            event.frame.save(frame_path)
        else:
            Image.fromarray(event.frame).save(frame_path)
        return frame_path
    return None


def process_tasks(
    worker_id: int,
    x_display: int,
    start_idx: int,
    end_idx: int,
    data_dir: str,
    raw_dir: str,
    img_dir: str,
    task_prefix: str,
    config_path: str,
    pass_list: str,
    fail_list: str
):
    """
    Worker function to process a range of tasks.

    Args:
        worker_id: ID of this worker process
        x_display: X display number to use
        start_idx: Starting task index (inclusive)
        end_idx: Ending task index (inclusive)
        data_dir: Directory containing task data
        raw_dir: Directory containing raw trajectory data
        img_dir: Directory to save images
        task_prefix: Prefix for task names
        config_path: Path to alfworld config file
        pass_list: Path to file for successful tasks
        fail_list: Path to file for failed tasks
    """
    print(f"[Worker {worker_id}] Starting with display :{x_display}, tasks {start_idx}-{end_idx}")

    # Set DISPLAY environment variable for this process
    os.environ['DISPLAY'] = f':{x_display}'

    # Load config
    with open(config_path) as reader:
        config = yaml.safe_load(reader)

    # Initialize environment for this worker
    env = ThorEnv(x_display=str(x_display))
    args = MockArgument()

    for i in range(start_idx, end_idx + 1):
        traj_name = f"{task_prefix}_{i}"

        # Skip if already processed
        if (Path(img_dir) / traj_name).is_dir():
            print(f"[Worker {worker_id}] Folder {traj_name} exists, skipping...")
            continue

        try:
            tw_file = os.path.join(data_dir, f"{traj_name}.traj.json")
            with open(tw_file) as f:
                tw_data = json.load(f)

            traj_file = os.path.join(raw_dir, tw_data["task_dir"], "traj_data.json")
            with open(traj_file) as f:
                traj_data = json.load(f)

            if img_dir:
                worker_img_dir = os.path.join(img_dir, traj_name)
                os.makedirs(worker_img_dir, exist_ok=True)

            traj_root = os.path.dirname(traj_file)
            task_desc = tw_data["task_desc"]
            turker_task_desc = traj_data['turk_annotations']['anns'][0]['task_desc']
            print(f"[Worker {worker_id}] Task: {task_desc} | {turker_task_desc}")

            # Initialize scene
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']
            scene_name = 'FloorPlan%d' % scene_num

            env.reset(scene_name)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            # Save initial state before OracleAgent explores
            initial_agent_state = {
                'position': env.last_event.metadata['agent']['position'].copy(),
                'rotation': env.last_event.metadata['agent']['rotation'].copy(),
                'horizon': env.last_event.metadata['agent']['cameraHorizon']
            }

            # Initialize oracle controller
            goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
            load_receps = config['controller']['load_receps']
            debug = True

            controller = OracleAgent(env, traj_data, tw_data, traj_root,
                                    load_receps=load_receps, debug=debug,
                                    goal_desc_human_anns_prob=goal_desc_human_anns_prob)

            print(f"[Worker {worker_id}] Initial state: {controller.intro}")

            # Get explored receptacles after initial exploration
            explored_receps = controller.get_explored_receps()
            tw_data['explored_receps'] = explored_receps

            # Restore agent to exact initial position
            env.step({
                'action': 'TeleportFull',
                'x': initial_agent_state['position']['x'],
                'y': initial_agent_state['position']['y'],
                'z': initial_agent_state['position']['z'],
                'rotateOnTeleport': False,
                'rotation': initial_agent_state['rotation'],
                'horizon': initial_agent_state['horizon'],
            })

            event = env.step(dict(traj_data['scene']['init_action']))
            env.set_task(traj_data, args, reward_type='dense')

            # Initialize trajectory
            tw_trajectory = []
            tw_trajectory.append({
                "name": "state_text",
                "content": controller.intro
            })

            # Save initial frame
            if img_dir:
                frame_path = save_frame(event, worker_img_dir, 'step_0')
                if frame_path:
                    tw_trajectory.append({
                        "name": "state_image",
                        "content": frame_path
                    })
                    
            # Execute actions
            for t, action_str in enumerate(tw_data["actions"]):
                feedback = controller.step(action_str)
                event = env.last_event
                print(f"[Worker {worker_id}] step: {t}, action: {action_str}, feedback: {feedback}")

                tw_trajectory.append({
                    "name": "action",
                    "content": action_str
                })
                tw_trajectory.append({
                    "name": "state_text",
                    "content": feedback
                })

                if img_dir:
                    frame_path = save_frame(event, worker_img_dir, f'step_{t+1}')
                    if frame_path:
                        tw_trajectory.append({
                            "name": "state_image",
                            "content": frame_path
                        })

                if not event.metadata['lastActionSuccess']:
                    print(f"[Worker {worker_id}] ERROR: {event.metadata['errorMessage']}")

            # Check goal satisfaction
            goal_satisfied = env.get_goal_satisfied()
            print(f"[Worker {worker_id}] goal_satisfied: {goal_satisfied}")

            tw_data["goal_satisfied"] = goal_satisfied
            tw_data["trajectory"] = tw_trajectory
            with open(tw_file, 'w') as f:
                json.dump(tw_data, f, indent=2)

            # Log results (using safe file locking to prevent race conditions)
            if not goal_satisfied:
                safe_append_to_file(fail_list, f"{traj_name}\n")
            else:
                safe_append_to_file(pass_list, f"{traj_name}\n")

        except FileNotFoundError as e:
            print(f"[Worker {worker_id}] File not found for {traj_name}: {e}")
            continue

        except Exception as e:
            print(f"[Worker {worker_id}] Exception during replay of {traj_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Clean shutdown
    print(f"[Worker {worker_id}] Shutting down environment...")
    try:
        env.stop()
    except (ProcessLookupError, OSError):
        pass
    print(f"[Worker {worker_id}] Done!")


def split_range(total_start: int, total_end: int, num_workers: int) -> List[Tuple[int, int]]:
    """
    Split a range into roughly equal chunks for parallel processing.

    Args:
        total_start: Starting index (inclusive)
        total_end: Ending index (inclusive)
        num_workers: Number of workers to split across

    Returns:
        List of (start, end) tuples for each worker
    """
    total_tasks = total_end - total_start + 1
    tasks_per_worker = total_tasks // num_workers
    remainder = total_tasks % num_workers

    ranges = []
    current_start = total_start

    for i in range(num_workers):
        # Distribute remainder across first workers
        worker_tasks = tasks_per_worker + (1 if i < remainder else 0)
        current_end = current_start + worker_tasks - 1
        ranges.append((current_start, current_end))
        current_start = current_end + 1

    return ranges


def main():
    # ===== Configuration =====
    NUM_WORKERS = 8          # Number of parallel processes
    START_DISPLAY = 0        # Starting X display number
    TOTAL_START = 91         # Starting task number
    TOTAL_END = 790           # Ending task number (inclusive)

    DATA_DIR = "/root/data/alfworld/train/"
    RAW_DIR = "/root/data/alfworld/raw/train"
    IMG_DIR = "/root/data/alfworld/train_images/"
    TASK_PREFIX = "task1"
    CONFIG_PATH = "alfworld/config.yaml"
    PASS_LIST = "/root/data/alfworld/sft_info/pass_list.txt"
    FAIL_LIST = "/root/data/alfworld/sft_info/fail_list.txt"

    # Create necessary directories
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PASS_LIST), exist_ok=True)
    os.makedirs(os.path.dirname(FAIL_LIST), exist_ok=True)

    # Split task range across workers
    work_ranges = split_range(TOTAL_START, TOTAL_END, NUM_WORKERS)

    print("="*80)
    print(f"Starting parallel processing with {NUM_WORKERS} workers")
    print(f"Total tasks: {TOTAL_START} to {TOTAL_END}")
    print("Work distribution:")
    for i, (start, end) in enumerate(work_ranges):
        display_num = START_DISPLAY + i
        print(f"  Worker {i}: tasks {start}-{end} (display :{display_num})")
    print("="*80 + "\n")

    # Create processes
    processes = []
    for worker_id, (start_idx, end_idx) in enumerate(work_ranges):
        display_num = START_DISPLAY + worker_id
        p = mp.Process(
            target=process_tasks,
            args=(
                worker_id,
                display_num,
                start_idx,
                end_idx,
                DATA_DIR,
                RAW_DIR,
                IMG_DIR,
                TASK_PREFIX,
                CONFIG_PATH,
                PASS_LIST,
                FAIL_LIST,
            )
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nMain process interrupted. Terminating all workers...")
        for p in processes:
            p.terminate()
            p.join()

    print("\n" + "="*80)
    print("All workers completed!")
    print("="*80)


if __name__ == "__main__":
    main()
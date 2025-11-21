import json
import os
import sys
import multiprocessing as mp
from typing import List, Tuple

# Add project root to Python path
# ROOT: meow_tea_gym/envs/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alfred.env.thor_env import ThorEnv
from PIL import Image


class MockArgument:
    """Fake argument class to reuse existing functions"""
    reward_config = 'alfred/env/rewards.json'


def get_abs_viewpose(env):
    """Snapshot the agent's exact pose as an absolute TeleportFull action."""
    a = env.last_event.metadata['agent']
    pos = a['position']
    yaw = a['rotation']['y'] % 360
    hor = a['cameraHorizon']
    return {
        'action': 'TeleportFull',
        'x': pos['x'], 'y': pos['y'], 'z': pos['z'],
        'rotation': yaw,
        'horizon': hor,
        'rotateOnTeleport': True,   # we use absolute pose
    }

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
    target_actions: List[str]
):
    """
    Worker function to process a range of tasks.

    Args:
        worker_id: ID of this worker process
        x_display: X display number to use (e.g., 0 for :0)
        start_idx: Starting task index (inclusive)
        end_idx: Ending task index (inclusive)
        data_dir: Directory containing task data
        raw_dir: Directory containing raw trajectory data
        img_dir: Directory to save images (optional, for debugging)
        task_prefix: Prefix for task names (e.g., "task1")
        target_actions: List of actions to track locations for
        pass_list: Path to file for successful tasks
        fail_list: Path to file for failed tasks
    """
    print(f"[Worker {worker_id}] Starting with display :{x_display}, tasks {start_idx}-{end_idx}")

    # Set DISPLAY environment variable for this process
    os.environ['DISPLAY'] = f':{x_display}'

    # Initialize environment for this worker
    env = ThorEnv(x_display=str(x_display))
    args = MockArgument()

    for i in range(start_idx, end_idx + 1):
        traj_name = f"{task_prefix}_{i}"

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

            print(f"[Worker {worker_id}] Processing {traj_name}...")

            # Initialize the scene and agent from the task info
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']

            scene_name = 'FloorPlan%d' % scene_num
            goal_instr = traj_data['turk_annotations']['anns'][0]['task_desc']

            env.reset(scene_name)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            event = env.step(dict(traj_data['scene']['init_action']))  # init action
            env.set_task(traj_data, args, reward_type='dense')  # set task to get reward

            # Optional: save initial frame for debugging
            if img_dir:
                save_frame(event, worker_img_dir, 'step_init')

            pos_data = []

            print(f"[Worker {worker_id}] Task Type: {traj_data['task_type']}")
            print(f"[Worker {worker_id}] Goal: {goal_instr}")
            print(f"[Worker {worker_id}] Scene: {traj_data['scene']['floor_plan']}")

            for t, ll_action in enumerate(traj_data['plan']['low_actions']):
                hl_action_idx, traj_api_cmd = ll_action['high_idx'], ll_action['api_action']

                if traj_api_cmd['action'] in target_actions:
                    # Capture the agent's viewpose for target actions
                    abs_viewpose = get_abs_viewpose(env)
                    pos_data.append({
                        "action": traj_api_cmd['action'],
                        "locs": abs_viewpose,
                        'objectId': traj_api_cmd['objectId'],
                        'receptacleObjectId': traj_api_cmd.get('receptacleObjectId', None)
                    })

                event = env.step(traj_api_cmd)

                # Optional: save frame for debugging
                if img_dir:
                    save_frame(event, worker_img_dir, f'step_{t:06d}')
                    
                t_reward, t_done = env.get_transition_reward()
                print(f"[Worker {worker_id}] step: {t}, action: {traj_api_cmd['action']}, "
                      f"t_reward: {t_reward}, t_success: {event.metadata['lastActionSuccess']}, t_done: {t_done}")

                if not event.metadata['lastActionSuccess']:
                    print(f"[Worker {worker_id}] ERROR: {event.metadata['errorMessage']}")

                if t_done:
                    break

            # Check if goal was satisfied
            goal_satisfied = env.get_goal_satisfied()
            print(f"[Worker {worker_id}] goal_satisfied: {goal_satisfied}")

            if goal_satisfied:
                print(f"[Worker {worker_id}] Goal Reached for {traj_name}")
                tw_data["precomputed_locs"] = pos_data
                with open(tw_file, 'w') as f:
                    json.dump(tw_data, f, indent=2)
                    

        except FileNotFoundError as e:
            print(f"[Worker {worker_id}] File not found for {traj_name}: {e}")
            continue

        except Exception as e:
            print(f"[Worker {worker_id}] Exception during processing {traj_name}: {e}")
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
    NUM_WORKERS = 6      # Number of parallel processes
    START_DISPLAY = 0    # Starting X display number
    TOTAL_START = 1    # Starting task number
    TOTAL_END = 308      # Ending task number (inclusive)

    DATA_DIR = "/root/data/alfworld/train/"
    RAW_DIR = "/root/data/alfworld/raw/train"
    IMG_DIR = "/root/data/alfred/train_images" # None to skip
    TASK_PREFIX = "task3"

    if TASK_PREFIX == "task1":
        TARGET_ACTIONS = ["PickupObject", "PutObject"]
    elif TASK_PREFIX == "task2":
        TARGET_ACTIONS = ["PickupObject", "PutObject", "ToggleObjectOn"]
    elif TASK_PREFIX == "task3":
        TARGET_ACTIONS = ["PickupObject", "PutObject", "ToggleObjectOn", "ToggleObjectOff"]

    

    # Split task range across workers
    work_ranges = split_range(TOTAL_START, TOTAL_END, NUM_WORKERS)

    print("="*80)
    print(f"Starting parallel processing with {NUM_WORKERS} workers")
    print(f"Total tasks: {TOTAL_START} to {TOTAL_END}")
    print(f"Target actions: {TARGET_ACTIONS}")
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
                display_num,  # Assign display number starting from START_DISPLAY
                start_idx,
                end_idx,
                DATA_DIR,
                RAW_DIR,
                IMG_DIR,
                TASK_PREFIX,
                TARGET_ACTIONS,
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

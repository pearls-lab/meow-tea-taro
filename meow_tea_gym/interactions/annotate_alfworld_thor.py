import sys
from pathlib import Path

# Add envs directory to Python path so alfred and alfworld can be imported
envs_path = Path(__file__).parent.parent / 'envs'
sys.path.insert(0, str(envs_path))

import json
import yaml
import os
from PIL import Image

from ..envs.alfred.env.thor_env import ThorEnv # type: ignore
from ..envs.alfworld.agents.controller import OracleAgent # type: ignore


class MockArgument:
    """Fake argument class to reuse existing functions"""
    reward_config = 'meow_tea_gym/envs/alfworld/rewards.json'


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


def print_ground_truth_actions(traj_data):
    """Print ground truth low-level actions for reference"""
    print("\n" + "="*80)
    print("GROUND TRUTH LOW-LEVEL ACTIONS (for reference):")
    print("="*80)

    for t, ll_action in enumerate(traj_data['plan']['low_actions']):
        api_action = ll_action['api_action']
        action_type = api_action['action']

        # Format the action nicely
        if action_type in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
                          'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']:
            obj_id = api_action.get('objectId', 'N/A')
            # Extract just the object type from the full ID
            obj_type = obj_id.split('|')[0] if '|' in obj_id else obj_id
            print(f"  {t+1:3d}. {action_type:20s} - {obj_type}")
        else:
            # Movement actions
            print(f"  {t+1:3d}. {action_type}")

    print("="*80 + "\n")


def annotate_task(task_id: int, x_display: int, data_dir: str, raw_dir: str,
                 img_dir: str, task_prefix: str, config_path: str, max_steps: int = 100,
                 annotated_list: str = None) -> bool:
    """
    Interactive annotation of a single task.

    Args:
        task_id: Task ID to annotate
        x_display: X display number to use
        data_dir: Directory containing task data
        raw_dir: Directory containing raw trajectory data
        img_dir: Directory to save annotation images
        task_prefix: Prefix for task names
        config_path: Path to alfworld config file

    Returns:
        bool: True if task was successfully annotated and goal satisfied
    """
    traj_name = f"{task_prefix}_{task_id}"

    print("\n" + "="*80)
    print(f"ANNOTATING TASK: {traj_name}")
    print("="*80)

    # Set DISPLAY environment variable
    os.environ['DISPLAY'] = f':{x_display}'

    # Load configuration
    with open(config_path) as reader:
        config = yaml.safe_load(reader)

    # Load task data
    tw_file = os.path.join(data_dir, f"{traj_name}.traj.json")
    with open(tw_file) as f:
        tw_data = json.load(f)

    traj_file = os.path.join(raw_dir, tw_data["task_dir"], "traj_data.json")
    with open(traj_file) as f:
        traj_data = json.load(f)

    traj_root = os.path.dirname(traj_file)
    task_desc = tw_data["task_desc"]
    turker_task_desc = traj_data['turk_annotations']['anns'][0]['task_desc']

    # Print task information
    print(f"\nTask Description: {task_desc}")
    print(f"Turker Description: {turker_task_desc}")

    # Print ground truth actions for reference
    print_ground_truth_actions(traj_data)

    # Initialize environment
    env = ThorEnv(x_display=str(x_display))
    args = MockArgument()

    try:
        # Setup scene
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

        # Initialize controller to get initial state description
        goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
        load_receps = config['controller']['load_receps']
        debug = True

        controller = OracleAgent(env, traj_data, tw_data, traj_root,
                                load_receps=load_receps, debug=debug,
                                goal_desc_human_anns_prob=goal_desc_human_anns_prob)

        initial_state_text = controller.intro

        # Restore agent to exact initial position after exploration
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

        # Create image directory
        task_img_dir = os.path.join(img_dir, traj_name)
        os.makedirs(task_img_dir, exist_ok=True)

        # Function to reset to initial state
        def reset_to_initial_state():
            """Reset environment and controller to initial state"""
            # Reset scene
            env.reset(scene_name)
            env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            
            # Restore initial position
            env.step({
                'action': 'TeleportFull',
                'x': initial_agent_state['position']['x'],
                'y': initial_agent_state['position']['y'],
                'z': initial_agent_state['position']['z'],
                'rotateOnTeleport': False,
                'rotation': initial_agent_state['rotation'],
                'horizon': initial_agent_state['horizon'],
            })
            
            # Re-initialize controller
            nonlocal controller
            controller = OracleAgent(env, traj_data, tw_data, traj_root,
                                    load_receps=load_receps, debug=debug,
                                    goal_desc_human_anns_prob=goal_desc_human_anns_prob)
            
            # Restore to initial position again after controller exploration
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
            
            return event

        # Save and display initial state
        print(f"\nInitial State:\n{initial_state_text}\n")
        initial_frame_path = save_frame(event, task_img_dir, 'step_0')
        if initial_frame_path:
            print(f"Initial image: {initial_frame_path}\n")

        # Interactive annotation loop
        step_num = 0
        goal_satisfied = False
        annotated_actions = []
        annotated_trajectory = [
            {"name": "state_text", "content": initial_state_text}
        ]
        if initial_frame_path:
            annotated_trajectory.append({"name": "state_image", "content": initial_frame_path})

        print("="*80)
        print("START ANNOTATION (type 'quit' to exit, 'help' for commands)")
        print("="*80)

        while step_num < max_steps:
            # Prompt user for action
            print(f"\n--- Step {step_num + 1} ---")
            action_str = input("Enter action: ").strip()

            if action_str.lower() in ['quit', 'exit', 'q']:
                print("\nAnnotation session ended by user.")
                break

            if action_str.lower() in ['redo', 'r']:
                print("\n" + "="*80)
                print("REDOING TASK - Resetting to initial state...")
                print("="*80)
                
                # Reset environment and controller
                event = reset_to_initial_state()
                
                # Reset tracking variables
                step_num = 0
                goal_satisfied = False
                annotated_actions = []
                annotated_trajectory = [
                    {"name": "state_text", "content": initial_state_text}
                ]
                if initial_frame_path:
                    annotated_trajectory.append({"name": "state_image", "content": initial_frame_path})
                
                print(f"\nInitial State:\n{initial_state_text}\n")
                print("Task reset complete. Starting over...")
                continue

            if action_str.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Type any valid action (e.g., 'go to fridge 1', 'open fridge 1')")
                print("  - 'redo' or 'r': Reset and start over from the beginning")
                print("  - 'quit' or 'q': Exit annotation")
                print("  - 'help': Show this help message")
                continue

            if not action_str:
                print("Empty action. Please enter a valid action.")
                continue

            # Execute action
            feedback = controller.step(action_str)
            event = env.last_event
            step_num += 1

            # Save action and feedback
            annotated_actions.append(action_str)
            annotated_trajectory.append({"name": "action", "content": action_str})
            annotated_trajectory.append({"name": "state_text", "content": feedback})

            # Print feedback
            print(f"\nFeedback: {feedback}")

            # Save and display image
            frame_path = save_frame(event, task_img_dir, f'step_{step_num}')
            if frame_path:
                annotated_trajectory.append({"name": "state_image", "content": frame_path})
                print(f"Image saved: {frame_path}")

            # Check for errors
            if not event.metadata['lastActionSuccess']:
                print(f"ERROR: {event.metadata['errorMessage']}")

            # Check if goal is satisfied
            goal_satisfied = env.get_goal_satisfied()
            if goal_satisfied:
                print("\n" + "="*80)
                print("GOAL SATISFIED! Task completed successfully!")
                print("="*80)
                print("\nFull annotated action sequence:")
                for idx, action in enumerate(annotated_actions, 1):
                    print(f"  {idx}. {action}")
                print()
                break

        # If goal was satisfied, save to traj file
        if goal_satisfied:
            print("\nSaving annotated data to traj file...")
            tw_data['annotated_actions'] = annotated_actions
            tw_data['annotated_trajectory'] = annotated_trajectory

            with open(tw_file, 'w') as f:
                json.dump(tw_data, f, indent=2)

            if annotated_list:
                with open(annotated_list, 'a') as f:
                    f.write(f"{traj_name}\n")

            print(f"✓ Annotated data saved to: {tw_file}")
            print(f"  - annotated_actions: {len(annotated_actions)} actions")
            print(f"  - annotated_trajectory: {len(annotated_trajectory)} entries")
        else:
            print("\nGoal not satisfied. Annotated data NOT saved to traj file.")
            print("(You can re-annotate this task later)")

        print(f"\nTotal steps: {step_num}")
        print(f"Goal satisfied: {goal_satisfied}")

        return goal_satisfied

    except KeyboardInterrupt:
        print("\n\nAnnotation interrupted by user (Ctrl+C)")
        return False

    except Exception as e:
        print(f"\nException during annotation: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean shutdown
        print("\nShutting down environment...")
        try:
            env.stop()
        except (ProcessLookupError, OSError):
            pass


def main():
    """Main function for interactive task annotation."""
    # ===== Configuration =====
    X_DISPLAY = 0                    # X display number to use

    DATA_DIR = "/root/data/alfworld/train/"
    RAW_DIR = "/root/data/alfworld/raw/train"
    IMG_DIR = "/root/data/alfworld/annotations/images/"
    TASK_PREFIX = "task1"
    CONFIG_PATH = "meow_tea_gym/envs/alfworld/config.yaml"
    FAIL_LIST = "/root/data/alfworld/sft_info/fail_list.txt"
    ANNOTATED_LIST = "/root/data/alfworld/sft_info/annotated_list.txt"

    # Create necessary directories
    os.makedirs(IMG_DIR, exist_ok=True)

    # Read fail list
    if not os.path.exists(FAIL_LIST):
        print(f"Error: Fail list not found at {FAIL_LIST}")
        print("Please run replay_gt_parallel.py first to generate the fail list.")
        return

    with open(FAIL_LIST, 'r') as f:
        failed_tasks = [line.strip() for line in f if line.strip()]

    if not failed_tasks:
        print("No failed tasks found in fail list!")
        return

    # Read already annotated tasks
    already_annotated = set()
    if os.path.exists(ANNOTATED_LIST):
        with open(ANNOTATED_LIST, 'r') as f:
            already_annotated = set(line.strip() for line in f if line.strip())

    # Filter out already annotated tasks
    tasks_to_annotate = [task for task in failed_tasks if task not in already_annotated]

    print("="*80)
    print("ALFWorld Task Annotation Tool")
    print("="*80)
    print(f"\nFound {len(failed_tasks)} failed tasks")
    print(f"Already annotated: {len(already_annotated)} tasks")
    print(f"Remaining to annotate: {len(tasks_to_annotate)} tasks")
    print(f"Using X display :{X_DISPLAY}")
    print(f"Images will be saved to: {IMG_DIR}")
    print("\n" + "="*80)

    if not tasks_to_annotate:
        print("\nAll failed tasks have already been annotated!")
        return

    # Annotate tasks one by one
    annotated_count = 0

    for task_name in tasks_to_annotate:
        # Extract task ID from task name (e.g., "task1_123" -> 123)
        try:
            task_id = int(task_name.split('_')[-1])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse task ID from '{task_name}', skipping...")
            continue

        print(f"\n\nNext task: {task_name} ({annotated_count + 1}/{len(failed_tasks)})")
        response = input("Annotate this task? (y/n/quit): ").strip().lower()

        if response == 'quit' or response == 'q':
            print("\nExiting annotation tool...")
            break

        if response != 'y' and response != 'yes':
            print("Skipping task...")
            continue

        # Annotate the task
        success = annotate_task(
            task_id=task_id,
            x_display=X_DISPLAY,
            data_dir=DATA_DIR,
            raw_dir=RAW_DIR,
            img_dir=IMG_DIR,
            task_prefix=TASK_PREFIX,
            config_path=CONFIG_PATH,
            max_steps=100,
            annotated_list=ANNOTATED_LIST
        )

        if success:
            annotated_count += 1
            print(f"\n✓ Successfully annotated {task_name}!")
        else:
            print(f"\n✗ Failed to complete annotation for {task_name}")

    # Summary
    print("\n" + "="*80)
    print("ANNOTATION SESSION COMPLETE")
    print("="*80)
    print(f"Successfully annotated: {annotated_count}/{len(failed_tasks)} tasks")
    print("="*80)


if __name__ == "__main__":
    main()

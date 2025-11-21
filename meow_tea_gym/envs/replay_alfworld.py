import json
from signal import signal
import yaml
import os
from PIL import Image
from pathlib import Path

from alfred.env.thor_env import ThorEnv # type: ignore
from alfworld.agents.controller import OracleAgent # type: ignore

NUM_DATA = 1
START = 719
DATA_DIR = "/root/data/alfworld/train/"
RAW_DIR = "/root/data/alfworld/raw/train"
IMG_DIR = "/root/data/alfworld/train_images/"
TASK_PREFIX = "task1"
PASS_LIST = "/root/data/alfworld/sft_info/pass_list.txt"
FAIL_LIST = "/root/data/alfworld/sft_info/fail_list.txt"
os.makedirs(IMG_DIR, exist_ok=True)

with open("alfworld/config.yaml") as reader:
    config = yaml.safe_load(reader)

class MockArgument:
    """Fake argument class to reuse exisiting functions"""
    reward_config = 'alfworld/rewards.json'

# Initialize environment once
env = ThorEnv(x_display='0')

tasks_list = []
with open("/root/task_ids_with_changes.json", "r") as f:
    tasks_list = json.load(f)
print(len(tasks_list))

# for i in range(START, START + NUM_DATA):
for i in range(len(tasks_list)):
    # traj_name = f"{TASK_PREFIX}_{i}"
    traj_name = tasks_list[i]
    print(f"Processing trajectory: {traj_name}")

    # if (Path(IMG_DIR) / traj_name).is_dir():
    #     print(f"Folder {traj_name} exists, skipping...")
    #     continue
    with open("/root/data/alfworld/sft_info/pick_debug.txt", "a") as f:
        f.write(f"Processing trajectory: {traj_name}\n")
    
    tw_file = os.path.join(DATA_DIR, f"{traj_name}.traj.json")
    with open(tw_file) as f:
        tw_data = json.load(f)
    
    traj_file = os.path.join(RAW_DIR, tw_data["task_dir"], "traj_data.json")
    with open(traj_file) as f:
        traj_data = json.load(f)

    traj_root = os.path.dirname(traj_file)
    task_desc = tw_data["task_desc"]
    turker_task_desc = traj_data['turk_annotations']['anns'][0]['task_desc']
    print(f"Task: {task_desc} | {turker_task_desc}")

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

    controller_type = config['controller']['type']
    goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
    load_receps =   config['controller']['load_receps']
    # debug = config['controller']['debug']
    debug = True

    controller = OracleAgent(env, traj_data, tw_data, traj_root,
                            load_receps=load_receps, debug=debug,
                            goal_desc_human_anns_prob=goal_desc_human_anns_prob)

    print(f"Initial state: {controller.intro}")
    # Get explored receptacles after initial exploration
    explored_receps = controller.get_explored_receps()
    tw_data['explored_receps'] = explored_receps

    # env.reset(scene_name)
    # env.restore_scene(object_poses, object_toggles, dirty_and_empty)

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

    # class args: pass
    # args.reward_config = 'alfworld/rewards.json'
    args = MockArgument()
    env.set_task(traj_data, args, reward_type='dense') 

    tw_trajectory = []
    tw_trajectory.append({
        "name": "state_text",
        "content": controller.intro 
    })

    os.makedirs(os.path.join(IMG_DIR, traj_name), exist_ok=True)
    if hasattr(event, 'frame') and event.frame is not None: # save initial frame
        frame_path = os.path.join(IMG_DIR, traj_name, f'step_0.png')
        if isinstance(event.frame, Image.Image):
            event.frame.save(frame_path)
        else:
            Image.fromarray(event.frame).save(frame_path)
        tw_trajectory.append({
            "name": "state_image",
            "content": frame_path
        })

    try:
        for t, action_str in enumerate(tw_data["actions"]):
        # for t, ll_action in enumerate(traj_data["plan"]["low_actions"]):
            # traj_api_cmd = ll_action['api_action']

            feedback = controller.step(action_str)
            event = env.last_event
            print(f"step: {t}, action: {action_str}, feedback: {feedback}")
            tw_trajectory.append({
                "name": "action",
                "content": action_str
            })
            tw_trajectory.append({
                "name": "state_text",
                "content": feedback
            })

            if hasattr(event, 'frame') and event.frame is not None:
                frame_path = os.path.join(IMG_DIR, traj_name, f'step_{t+1}.png')
                if isinstance(event.frame, Image.Image):
                    event.frame.save(frame_path)
                else:
                    Image.fromarray(event.frame).save(frame_path)
                tw_trajectory.append({
                    "name": "state_image",
                    "content": frame_path
                })

            if not event.metadata['lastActionSuccess']:
                print(f"\t\terror: {event.metadata['errorMessage']}")

        goal_satisfied = env.get_goal_satisfied()
        print(f"goal_satisfied: {goal_satisfied}")
        tw_data["goal_satisfied"] = goal_satisfied
        tw_data["trajectory"] = tw_trajectory
        with open(tw_file, 'w') as f:
            json.dump(tw_data, f, indent=2)

        if not goal_satisfied:
            with open(FAIL_LIST, "a") as f:
                f.write(f"{traj_name}\n")
        else:
            with open(PASS_LIST, "a") as f:
                f.write(f"{traj_name}\n")

    except Exception as e:
        print(f"Exception during replay of {traj_file}: {e}")

# Clean shutdown
try:
    env.stop()
except (ProcessLookupError, OSError):
    pass
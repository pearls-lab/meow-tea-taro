import json
import yaml
import os
from PIL import Image
from pathlib import Path

from alfworld.env.thor_env_v5 import ThorEnv # type: ignore
from alfworld.agents.controller import OracleAgent # type: ignore

DATA_LEN = 6374
DATA_DIR = "/root/data/alfworld/train_v5/"
PDDL_DIR = "/root/data/alfworld_tw/train/"
IMG_DIR = "/root/data/alfworld/train_v5_action_images/"
os.makedirs(IMG_DIR, exist_ok=True)

with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

# Initialize environment once
env = ThorEnv(scene="FloorPlan1")

traj_name = f"train_33"
print(traj_name)

traj_file = f"{traj_name}.json"

pddl_file = f"{traj_name}.tw-pddl"

with open(PDDL_DIR + pddl_file) as f:
    pddl_data = json.load(f)
    walkthrough = pddl_data.get("walkthrough", [])

traj_root = os.path.dirname(traj_file)
with open(DATA_DIR + traj_file) as f:
    traj_data = json.load(f)

task_desc = traj_data['turk_annotations']['anns'][0]['task_desc']
print(f"Task: {task_desc}")
print(walkthrough)

scene_num = traj_data['scene']['scene_num']
object_poses = traj_data['scene']['object_poses']
dirty_and_empty = traj_data['scene']['dirty_and_empty']
object_toggles = traj_data['scene']['object_toggles']
scene_name = 'FloorPlan%d' % scene_num
env.reset(scene_name)
env.restore_scene(traj_data, object_poses, object_toggles, dirty_and_empty)
event = env.step(dict(traj_data['scene']['init_action']))

class args: pass
args.reward_config = 'rewards.json'
env.set_task(traj_data, args)

os.makedirs(os.path.join(IMG_DIR, traj_name), exist_ok=True)
if hasattr(event, 'frame') and event.frame is not None: # save initial frame
    frame_path = os.path.join(IMG_DIR, traj_name, f'step_0.png')
    if isinstance(event.frame, Image.Image):
        event.frame.save(frame_path)
    else:
        Image.fromarray(event.frame).save(frame_path)

controller_type = config['controller']['type']
goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
load_receps =   config['controller']['load_receps']
debug = config['controller']['debug']

controller = OracleAgent(env, traj_data, traj_root,
                        load_receps=load_receps, debug=debug,
                        goal_desc_human_anns_prob=goal_desc_human_anns_prob)

try:
    for t, action_str in enumerate(walkthrough):

        feedback = controller.step(action_str)
        event = env.last_event

        if hasattr(event, 'frame') and event.frame is not None:
            frame_path = os.path.join(IMG_DIR, traj_name, f'step_{t+1}.png')
            if isinstance(event.frame, Image.Image):
                event.frame.save(frame_path)
            else:
                Image.fromarray(event.frame).save(frame_path)

        if not event.metadata['lastActionSuccess']:
            print(f"\t\terror: {event.metadata['errorMessage']}")

    goal_satisfied = env.get_goal_satisfied()
    print(f"goal_satisfied: {goal_satisfied}")

except Exception as e:
    print(f"Exception during replay of {traj_file}: {e}")

env.stop()
import json
import yaml
import os
from PIL import Image
from pathlib import Path

from alfworld.env.thor_env_v5 import ThorEnv # type: ignore
from alfworld.agents.controller import OracleAgent # type: ignore


def find_correct_object(traj_object_id, event_metadata):
    """
    Find the correct object from the scene metadata.

    Args:
        traj_object_id: The objectId from trajectory (may be outdated)
    
    Returns:
        Correct objectId from the scene, or None if not found
    """
    objects = event_metadata.get('objects', [])
    
    # First, check if the exact objectId exists in the scene
    for obj in objects:
        if obj.get('objectId') == traj_object_id:
            return obj
    
    # Extract object name/type from objectId (format: "Type|x|y|z")
    object_name = traj_object_id.split('|')[0] if '|' in traj_object_id else traj_object_id
    
    # Try to find by exact name match (case-insensitive)
    candidates = []
    for obj in objects:
        obj_name = obj.get('name', '')
        obj_type = obj.get('objectType', '')
        obj_id = obj.get('objectId', '')
        
        # Check if name or type matches
        if obj_name.lower() == object_name.lower() or obj_type == object_name:
            # Prefer visible objects
            if obj.get('visible', False):
                return obj
            candidates.append(obj)
    
    # If found candidates but none visible, return the first one
    if candidates:
        return candidates[0]
    
    # If still not found, try partial name match
    for obj in objects:
        obj_name = obj.get('name', '').lower()
        obj_id = obj.get('objectId', '')
        if object_name.lower() in obj_name or obj_name in object_name.lower():
            if obj.get('visible', False):
                return obj
            if not candidates:
                candidates.append(obj)
    
    return candidates[0] if candidates else None


def fix_object_ids_in_action(api_cmd, event_metadata):
    """
    Fix objectIds in the action command if they don't exist in the scene.
    
    Args:
        api_cmd: The API action command from trajectory
    
    Returns:
        Fixed action command with correct objectIds
    """
    fixed_cmd = api_cmd.copy()
    
    # Check for objectId
    if 'objectId' in fixed_cmd:
        #breakpoint()
        traj_obj_id = fixed_cmd['objectId']
        correct_obj = find_correct_object(traj_obj_id, event_metadata)
        if correct_obj:
            if correct_obj['objectId'] != traj_obj_id:
                print(f"  [Fix] objectId: {traj_obj_id} -> {correct_obj['objectId']}")
            fixed_cmd['objectId'] = correct_obj['objectId']
        else:
            print(f"  [Warning] Could not find correct objectId for: {traj_obj_id}")
    
    # Check for receptacleObjectId (used in PutObject)
    if 'receptacleObjectId' in fixed_cmd:
        traj_recep_id = fixed_cmd['receptacleObjectId']
        correct_recep = find_correct_object(traj_recep_id, event_metadata)
        if correct_recep:
            if correct_recep['objectId'] != traj_recep_id:
                print(f"  [Fix] receptacleObjectId: {traj_recep_id} -> {correct_recep['objectId']}")
            fixed_cmd['receptacleObjectId'] = correct_recep['objectId']
        else:
            print(f"  [Warning] Could not find correct receptacleObjectId for: {traj_recep_id}")
    
    fixed_cmd['forceAction'] = True  # Ensure forceAction is always True
    return fixed_cmd


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
# print(walkthrough)

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
    frame_path = os.path.join(IMG_DIR, traj_name, f'0000.png')
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
print(traj_data['plan']['low_actions'])

try:
    for t, ll_action in enumerate(traj_data['plan']['low_actions']):
        hl_action_idx, traj_api_cmd = ll_action['high_idx'], ll_action['api_action']
        # Fix objectIds before executing action
        fixed_cmd = fix_object_ids_in_action(traj_api_cmd, env.last_event.metadata)
        # We need this to pass the task
        fixed_cmd['forceAction'] = True
        event = env.step(fixed_cmd)

        if hasattr(event, 'frame') and event.frame is not None:
            frame_path = os.path.join(IMG_DIR, traj_name, f'{t+1:04d}.png')
            if isinstance(event.frame, Image.Image):
                event.frame.save(frame_path)
            else:
                Image.fromarray(event.frame).save(frame_path)

        t_reward, t_done = env.get_transition_reward()
        print(t_done)
        print(f"step: {t}, action: {fixed_cmd}, t_reward: {t_reward}, t_success: {event.metadata['lastActionSuccess']}, t_done: {t_done}")
        if not event.metadata['lastActionSuccess']:
            print(f"\t\terror: {event.metadata['errorMessage']}")

        if t_done:
            break

    goal_satisfied = env.get_goal_satisfied()
    print(f"goal_satisfied: {goal_satisfied}")

except Exception as e:
    print(f"Exception during replay of {traj_file}: {e}")

env.stop()
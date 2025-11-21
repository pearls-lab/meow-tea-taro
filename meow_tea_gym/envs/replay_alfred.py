import json
import os
import numpy as np
from alfred.env.thor_env import ThorEnv
from PIL import Image

os.environ['DISPLAY'] = ':0'

class MockArgument:
    """Fake argument class to reuse exisiting functions"""
    reward_config = 'alfred/env/rewards.json'


import math

def find_valid_interaction_point(env, object_id):
    """
    Manually searches for a valid teleport point where the object is visible.
    Compatible with AI2-THOR 2.1.0 (ALFRED version).
    """
    # 1. Find the object's ground truth position
    # We need to scan the last event's metadata to find the target object
    target_obj = None
    all_objects = env.last_event.metadata['objects']
    for obj in all_objects:
        if obj['objectId'] == object_id:
            target_obj = obj
            break
            
    if not target_obj:
        print(f"Check failed: Object {object_id} not found in scene metadata.")
        return None

    obj_pos = target_obj['position']
    
    # 2. Generate candidate standing points (Polar grid around object)
    # We search radius 0.5m to 1.5m (interaction distance is usually 1.5m)
    radii = [0.5, 0.75, 1.0, 1.25] 
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    print(f"Scanning for valid view of {object_id} at {obj_pos}...")

    for r in radii:
        for angle_deg in angles:
            angle_rad = math.radians(angle_deg)
            
            # Calculate candidate agent position
            # (X, Z) are the floor plane in Unity
            cand_x = obj_pos['x'] + r * math.sin(angle_rad)
            cand_z = obj_pos['z'] + r * math.cos(angle_rad)
            
            # Calculate Rotation to look AT the object
            # atan2(target - agent)
            dx = obj_pos['x'] - cand_x
            dz = obj_pos['z'] - cand_z
            look_yaw = math.degrees(math.atan2(dx, dz))
            
            # AI2-THOR expects 0-360
            if look_yaw < 0: look_yaw += 360
            
            # Snap to nearest 45 degrees (ALFRED restriction) or 90
            # ALFRED usually uses 90 increments, but 15/45 works for teleport
            # Let's try exact rotation first, or snap to 45
            look_yaw = round(look_yaw / 15) * 15 

            # 3. Teleport
            action = {
                "action": "TeleportFull",
                "x": cand_x,
                "y": target_obj['position']['y'], # Keep roughly same height logic (agent snaps to floor)
                "z": cand_z,
                "rotation": {"x": 0, "y": look_yaw, "z": 0},
                "horizon": 30, # Look down slightly (standard interaction)
                "standing": True,
                "forceAction": True
            }
            
            # Use the raw environment step
            event = env.step(action)
            
            if not event.metadata['lastActionSuccess']:
                continue # Hit a wall or obstacle
                
            # 4. Check Visibility
            # Find the object in the new metadata
            # In AI2-THOR 2.1.0, 'visible' means "in camera view + close enough + unobstructed"
            for obj in event.metadata['objects']:
                if obj['objectId'] == object_id:
                    if obj['visible']:
                        return action # SUCCESS: Found a working spot
                    break
                    
    return None # Failed to find any spot


if __name__ == "__main__":
    file_name = "/root/data/alfworld/raw/train/pick_and_place_simple-SoapBottle-None-SideTable-420/trial_T20190908_063550_595984/traj_data.json"
    img_dir = '/root/data/images/'
    os.makedirs(img_dir, exist_ok=True)
    
    success_cnt = 0

    with open(file_name) as f:
        traj_data = json.load(f)

    env = ThorEnv(x_display='0')

    args = MockArgument()

    # intialize the scene and agent from the task info
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    # NOTE: `turk_annotations` are only available in ALFRED (not for infini-thor :/ )
    goal_instr = traj_data['turk_annotations']['anns'][0]['task_desc']
    
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    event = env.step(dict(traj_data['scene']['init_action'])) # init action
    env.set_task(traj_data, args, reward_type='dense') # set task to get reward
    
    # For debugging: save the frame after each step
    if hasattr(event, 'frame') and event.frame is not None: # save initial frame
        frame_path = os.path.join(img_dir, f'init.png')
        if isinstance(event.frame, Image.Image):
            event.frame.save(frame_path)
        else:
            Image.fromarray(event.frame).save(frame_path)

    print(f"\t\tTask Type: {traj_data['task_type']}")
    print(f"\t\tGoal: {goal_instr}\n\t\tScene: {traj_data['scene']['floor_plan']}")

    for t, ll_action in enumerate(traj_data['plan']['low_actions']):
        hl_action_idx, traj_api_cmd = ll_action['high_idx'], ll_action['api_action']
        
        event = env.step(traj_api_cmd)
        # REPLACE THE RAW CONTROLLER BLOCK WITH THIS:
        
        target_id = "SoapBottle|-02.40|+01.05|-00.49"
        
        print(f"Step {t}: Searching for valid pose for {target_id}...")
        
        # Call our manual search function
        valid_action = find_valid_interaction_point(env, target_id)
        
        if valid_action:
            print(f"  SUCCESS: Found valid pose: {valid_action}")
            # The agent is now already standing there because the function 
            # executed the TeleportFull command during the check.
            
            # OPTIONAL: You can save the valid_action parameters to a file now
            # valid_poses.append(valid_action)
        else:
            print(f"  FAILURE: Could not find any reachable view of {target_id}")
        
    #     # For debugging: save frame
    #     if hasattr(event, 'frame') and event.frame is not None:
    #         frame_path = os.path.join(img_dir, f'step_{t:06d}.png')
    #         if isinstance(event.frame, Image.Image):
    #             event.frame.save(frame_path)
    #         else:
    #             Image.fromarray(event.frame).save(frame_path)
        
    #     t_reward, t_done = env.get_transition_reward()
    #     print(f"step: {t}, action: {traj_api_cmd}, t_reward: {t_reward}, t_success: {event.metadata['lastActionSuccess']}, t_done: {t_done}")
    #     if not event.metadata['lastActionSuccess']:
    #         print(f"\t\terror: {event.metadata['errorMessage']}")

    #     if t_done:
    #         break

    # # check if goal was satisfied
    # goal_satisfied = env.get_goal_satisfied()
    # print(f"goal_satisfied: {goal_satisfied}")
    # if goal_satisfied:
    #     print("Goal Reached")
    #     success_cnt += 1
    
print(f"Success count: {success_cnt}")

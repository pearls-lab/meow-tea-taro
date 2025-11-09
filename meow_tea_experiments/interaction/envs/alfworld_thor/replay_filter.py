import json
import yaml
import os
from PIL import Image

from alfworld.env.thor_env_v5 import ThorEnv # type: ignore
from alfworld.agents.controller import OracleAgent # type: ignore
from utils import fix_object_ids_in_action

DATA_LEN = 6374
DATA_DIR = "/root/data/alfworld/train/"
OUT_DIR = "/root/data/alfworld/train_v5/"
OUT_DISCARD_DIR = "/root/data/alfworld/train_discarded_v5/"
IMG_DIR = "/root/data/alfworld/images_train/"

with open("config.yaml") as reader:
    config = yaml.safe_load(reader)

# Initialize environment once
env = ThorEnv(scene="FloorPlan1")

for i in range(1, DATA_LEN+1):
    file = f"train_{i}.json"
    if file in os.listdir(OUT_DIR) or file in os.listdir(OUT_DISCARD_DIR):
        continue
    filename = f"{DATA_DIR}{file}"

    traj_root = os.path.dirname(filename)
    with open(filename) as f:
        traj_data = json.load(f)

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

    traj_name = f"train_{i}"
    os.makedirs(os.path.join(IMG_DIR, traj_name), exist_ok=True)
    if hasattr(event, 'frame') and event.frame is not None: # save initial frame
        frame_path = os.path.join(IMG_DIR, traj_name, f'{0:04d}.png')
        if isinstance(event.frame, Image.Image):
            event.frame.save(frame_path)
        else:
            Image.fromarray(event.frame).save(frame_path)

    # controller_type = config['controller']['type']
    # goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
    # load_receps =   config['controller']['load_receps']
    # debug = config['controller']['debug']

    # controller = OracleAgent(env, traj_data, traj_root,
    #                         load_receps=load_receps, debug=debug,
    #                         goal_desc_human_anns_prob=goal_desc_human_anns_prob)

    try:
        for t, ll_action in enumerate(traj_data['plan']['low_actions']):
            hl_action_idx, traj_api_cmd = ll_action['high_idx'], ll_action['api_action']
            
            # Fix objectIds before executing action
            fixed_cmd = fix_object_ids_in_action(env, traj_api_cmd, env.last_event.metadata)
            
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

        if goal_satisfied:
            out_path = os.path.join(OUT_DIR, f"train_{i}.json")
            with open(out_path, 'w') as out_f:
                json.dump(traj_data, out_f)
        else:
            print("Goal Not Reached")
            discard_path = os.path.join(OUT_DISCARD_DIR, f"train_{i}.json")
            with open(discard_path, 'w') as discard_f:
                json.dump(traj_data, discard_f)

    except Exception as e:
        print(f"Exception occurred while processing {filename}: {e}")
        # If an exception occurs, move the file to the discard directory
        discard_path = os.path.join(OUT_DISCARD_DIR, f"train_{i}.json")
        with open(discard_path, 'w') as discard_f:
            json.dump(traj_data, discard_f)
        continue

env.stop()
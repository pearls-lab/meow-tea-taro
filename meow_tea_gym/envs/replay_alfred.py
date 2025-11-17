import json
import os

from alfred.env.thor_env import ThorEnv
from PIL import Image

os.environ['DISPLAY'] = ':0'

class MockArgument:
    """Fake argument class to reuse exisiting functions"""
    reward_config = 'alfred/env/rewards.json'


if __name__ == "__main__":
    file_name = "/root/data/alfworld/raw/train/pick_and_place_simple-ButterKnife-None-Drawer-8/trial_T20190918_144242_057762/traj_data.json"
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
        
        # For debugging: save frame
        if hasattr(event, 'frame') and event.frame is not None:
            frame_path = os.path.join(img_dir, f'step_{t:06d}.png')
            if isinstance(event.frame, Image.Image):
                event.frame.save(frame_path)
            else:
                Image.fromarray(event.frame).save(frame_path)
        
        t_reward, t_done = env.get_transition_reward()
        print(f"step: {t}, action: {traj_api_cmd}, t_reward: {t_reward}, t_success: {event.metadata['lastActionSuccess']}, t_done: {t_done}")
        if not event.metadata['lastActionSuccess']:
            print(f"\t\terror: {event.metadata['errorMessage']}")

        if t_done:
            break

    # check if goal was satisfied
    goal_satisfied = env.get_goal_satisfied()
    print(f"goal_satisfied: {goal_satisfied}")
    if goal_satisfied:
        print("Goal Reached")
        success_cnt += 1
    
print(f"Success count: {success_cnt}")

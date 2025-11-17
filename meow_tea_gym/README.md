# The meow-tea-gym environment: ALFWorld (ThorEnv) 

## Efforts to fix ALFWorld simulator
### Problem
Why we need this: The ALFRED simulator (built on ai2thor) is a stable simulator. ALFWorld wraps the ALFRED simulator by creating a map of `actual object locations` to `named objects`. For example, the agent needs to `MoveAhead`, `RotateRight` several times to get to the **close enough location** to interact with the object. In ALFWorld, on the contratry, the built-in *oracle agent* first explores the scene and assign each object it sees with a number (like table 2, fork 1); then the agent just needs to do `go to table 2` to get teleported to the designated position of table 2.

The `scene exploration` of ALFWorld's oracle agent is not stable, unfortunately. Sometimes the agent cannot be successfully teleported to the correct location because either 1) the location of the object is not correctly recorded during exploration, or 2) the teleport action fails due to object collision (an object is in between the agent and the location).

Therefore, we put effort into fixing the ALFWorld's simulator so that the agent 1) can be teleported to a location successfully and 2) can interact smoothly with the receptacles, **just as in the textworld-version** of ALFWorld.

### Proposed solution: directly teleport the agent to precomputed location from ALFRED
Each ALFWorld task is associated with an ALFRED task. We collect the precomputed location of the specific objects referred in `PickupObject` and `PutObject` lower-level actions when replaying the ALFRED's trajectories. Then in ALFWorld, we directly teleport the agent to the precomputed location we recorded.

To run the precomputing locs function in parallel:

```bash
# Navigate to meow_tea_gym/envs directory (REQUIRED)
cd meow-tea-taro/meow_tea_gym/envs

# Run the script
python alfworld/utils/precompute_locs_parallel.py
```

**Modified files:**
- Updates `*.traj.json` files with `precomputed_locs` field containing:
  ```json
  {
    "action": "PickupObject",
    "locs": {
      "action": "TeleportFull",
      "x": 1.5, "y": 0.9, "z": 2.3,
      "rotation": 90,
      "horizon": 0,
      "rotateOnTeleport": true
    },
    "objectId": "Apple|+01.50|+00.90|+02.30",
    "receptacleObjectId": null
  }
  ```

**Example Output:**
```
================================================================================
Starting parallel processing with 8 workers
Total tasks: 91 to 790
Work distribution:
  Worker 0: tasks 91-178 (display :0)
  Worker 1: tasks 179-266 (display :1)
  ...
  Worker 7: tasks 666-790 (display :7)
================================================================================

[Worker 0] Starting with display :0, tasks 91-178
[Worker 0] Task: put a clean lettuce in fridge | Put a clean lettuce in the fridge
[Worker 0] Initial state: You are in the middle of a room...
[Worker 0] step: 0, action: go to fridge 1, feedback: The fridge 1 is closed.
...
[Worker 0] goal_satisfied: True
```

---

## Other useful scripts
### Replay
Replays ground-truth trajectories using the OracleAgent controller. Saves full trajectories with state observations, actions, and images.

```bash
# Navigate to meow_tea_gym/envs directory (REQUIRED)
cd meow-tea-taro/meow_tea_gym/envs

# Run the script
python alfworld/utils/replay_gt_parallel.py
```

**Modified files:**
- Updates `*.traj.json` files with:
  - `explored_receps`: Receptacles explored during initial navigation
  - `goal_satisfied`: Whether the goal was achieved
  - `trajectory`: Full trajectory with state_text, actions, and state_image paths

**Image directories:**
- Creates `{IMG_DIR}/{task_name}/` for each task
- Saves step images: `step_0.png`, `step_1.png`, etc.

**Success/Failure logs:**
- `PASS_LIST`: List of successfully completed tasks
- `FAIL_LIST`: List of failed tasks


### Annotation
Provides interactive annotation of failed tasks from the replay step. Annotators manually execute actions in the ALFWorld environment to create supervised fine-tuning (SFT) datasets with action sequences and trajectory images.

**Prerequisites:**
- Run `replay_gt_parallel.py` first to generate the fail list

**Usage:**

```bash
cd meow-tea-taro
python -m meow_tea_gym.interactions.annotate_alfworld_thor
```

**Available Commands:**
- Type action (e.g., `go to fridge 1`, `open fridge 1`, `pickup apple 1`)
- `redo` or `r`: Reset environment and restart annotation from beginning
- `quit` or `q`: Exit annotation session
- `help`: Show available commands

**Modified files:**
- Updates `*.traj.json` files with:
  - `annotated_actions`: List of annotator-provided action strings
  - `annotated_trajectory`: Full trajectory containing:
    - Initial state text and image
    - Each action executed
    - State text feedback after each action
    - State images after each action

**Output directories:**
- Creates `{IMG_DIR}/{task_name}/` for annotated images
- Saves images: `step_0.png` (initial), `step_1.png`, `step_2.png`, etc.
- Updates annotated_list.txt with successfully annotated task names


import cv2
import copy
import alfworld.gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from alfworld.env.tasks import get_task
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import alfworld.gen.utils.image_util as image_util
from alfworld.gen.utils import game_util
from alfworld.gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj


DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': False,
                           'renderClassImage': False,
                           'renderObjectImage': False,
                           }

class ThorEnv(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self,
                 scene,
                 x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 smooth_nav=False,
                 save_frames_to_disk=False,
                 save_frames_path="./",
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH,
                 platform=CloudRendering):

        super().__init__(scene=scene, quality=quality, platform=platform)
        self.local_executable_path = build_path
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # intermediate states for CoolObject Subgoal
        self.cooled_reward = False
        self.reopen_reward = False

        self.save_frames_to_disk = save_frames_to_disk
        self.save_frames_path = save_frames_path
        self.smooth_nav = smooth_nav

        print("ThorEnv started.")

    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num

        super().reset(scene_name)
        event = super().step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibility_distance=visibility_distance,
            makeAgentsVisible=False,
        ))

        # reset task if specified
        if hasattr(self, 'task'):
            if self.task is not None:
                self.task.reset()

        # clear object state changes
        self.reset_states()

        return event

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, traj, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibility_distance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))

        before_objs = self.last_event.metadata.get('objects', []) if self.last_event else []
        before_movable = [o for o in before_objs if o.get('pickupable') or o.get('moveable')]
        print(f"[Setup] Before placement: total={len(before_objs)}, movable={len(before_movable)}")
        
        object_poses = _overwrite_objects_from_traj(self, traj)

        # restore states or dirty_and_empty
        if dirty_and_empty:
            for o in object_poses:
                super().step(dict(action='SetObjectStates',
                    SetObjectStates={'objectType': o['objectName'].split('_')[0], 'stateChange': 'dirtyable', 'isDirty': True}))

                super().step(dict(action='SetObjectStates',
                    SetObjectStates={'objectType': o['objectName'].split('_')[0], 'stateChange': 'canFillWithLiquid', 'isFilledWithLiquid': False}))

        # e.g., object_toggles = [{'isOn': False, 'objectType': 'FloorLamp'}]
        # if object_toggles:
        #     for o in object_toggles:
                # super().step(dict(action='SetObjectStates',
                #     #SetObjectStates={'objectType': o['objectType'], 'stateChange': 'toggleable', 'isToggled': False})) # 'stateChange': 'toggleable' doesn't work
                #     SetObjectStates={'objectType': o['objectType'], 'isToggled': False}))
        if object_toggles:
            obj_toggle_types = [o['objectType'] for o in object_toggles]
            for o in self.last_event.metadata["objects"]:
                if o['objectType'] in obj_toggle_types and o['isToggled'] == True:
                    super().step(dict(action='ToggleObjectOff', objectId=o['objectId'], forceAction=True))

        # Diagnostics after placement
        after_objs = self.last_event.metadata.get('objects', []) if self.last_event else []
        after_movable = [o for o in after_objs if o.get('pickupable') or o.get('moveable')]
        print(f"[Setup] After placement: total={len(after_objs)}, movable={len(after_movable)}")

    def set_task(self, traj, args, reward_type='sparse', max_episode_length=2000):
        '''
        set the current task type (one of 7 tasks)
        '''
        task_type = traj['task_type']
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length)

    def step(self, action=None, smooth_nav=False, **action_args):
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if smooth_nav:
            if "MoveAhead" in action['action']:
                self.smooth_move_ahead(action)
            elif "Rotate" in action['action']:
                self.smooth_rotate(action)
            elif "Look" in action['action']:
                self.smooth_look(action)
            else:
                super().step(action)
        else:
            if "LookUp" in action:
                self.look_angle(-constants.AGENT_HORIZON_ADJ)
            elif "LookDown" in action:
                self.look_angle(constants.AGENT_HORIZON_ADJ)
            elif isinstance(action, dict):
                # compatible with ai2thor==5.0.0
                if action['action'] == 'TeleportFull':
                    if 'rotateOnTeleport' in action:
                        del action['rotateOnTeleport']
                    action["standing"] = True
                elif action['action'] == 'PutObject':
                    action['objectId'] = action['receptacleObjectId']
                    del action['receptacleObjectId']
                elif "ToggleObject" in action['action']:
                    if 'coordinateReceptacleObjectId' in action:
                        del action['coordinateReceptacleObjectId']
                    if 'forceVisible' in action:
                        del action['forceVisible']
                    if 'cleanObjectId' in action:
                        del action['cleanObjectId']
                    if 'coordinateObjectId' in action:
                        del action['coordinateObjectId'] 
                super().step(action)
            else:
                super().step(action=action, **action_args)
        
        event = self.update_states(action)
        self.check_post_conditions(action)
        return event

    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        # add 'cleaned' to all object that were washed in the sink
        event = self.last_event
        if event.metadata['lastActionSuccess']:
            action_type = action['action'] if isinstance(action, dict) else action
            # clean
            if action_type == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = game_util.get_obj_of_type_closest_to_obj(
                    'SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action_type == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = game_util.get_objects_of_type(
                    'Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action_type == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = game_util.get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()
        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.last_event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.last_event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.last_event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 90)
        else:
            end_rotation = (start_rotation + 90)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'standing': True
            #'tempRenderChange': True,
            #'renderNormalsImage': False,
            #'renderImage': render_settings['renderImage'],
            #'renderClassImage': render_settings['renderClassImage'],
            #'renderObjectImage': render_settings['renderObjectImage'],
            #'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.last_event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        # TODO: parametrized navigation commands

        if "RotateLeft" in action:
            action = dict(action="RotateLeft",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            action = dict(action="LookUp",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            action = dict(action="LookDown",
                          forceAction=True)
            event = self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            event = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            event = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            event = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=inventory_object_id,
                          receptacleObjectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            event = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            event = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            event = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.last_event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            event = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return event, action

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.last_event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            event = self.step({'action': 'Pass'})
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event.metadata)['isDirty']):
                    event = self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids

    def va_interact(self, action, interact_mask=None, smooth_nav=True, mask_px_sample=1, debug=False):
        '''
        interact mask based action call
        '''

        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(self.last_event.instance_segmentation_frame)
            color_to_object_id = self.last_event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            if debug:
                print("action_box", "instance_ids", instance_ids)

            # prune invalid instances like floors, walls, etc.
            instance_ids = self.prune_by_any_interaction(instance_ids)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                err = "Bad interact mask. Couldn't locate target object"
                success = False
                return success, None, None, err, None

            target_instance_id = instance_ids[0]
        else:
            target_instance_id = ""
        
        # if debug:
        #     print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        try:
            event, api_action = self.to_thor_api_exec(action, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            return success, None, None, err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.last_event.frame[:,:,::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            return success, event, target_instance_id, event.metadata['errorMessage'], api_action

        success = True
        return success, event, target_instance_id, '', api_action

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)


def _normalize_type_name(s: Optional[str]) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _object_name_pool_by_type(event) -> Dict[str, List[str]]:
    """Return mapping: normalized type -> list of object *names* (not ids) available in scene.
    Only includes pickupable/moveable objects, since SetObjectPoses operates on those.
    """
    pool: Dict[str, List[str]] = {}
    objs = event.metadata.get('objects', []) if event else []
    for o in objs:
        if not (o.get('pickupable') or o.get('moveable')):
            continue
        t = _normalize_type_name(o.get('objectType'))
        name = o.get('name')
        if not t or not name:
            continue
        pool.setdefault(t, []).append(name)
    return pool


def _build_objectposes_from_traj(traj: Dict[str, Any], name_pool: Dict[str, List[str]]):
    """Build a SetObjectPoses payload list from trajectory scene.object_poses.

    For each desired entry, we pick a *source* objectName from the current scene
    with the same type. If there are more desired instances than names in the pool,
    we reuse a name; SetObjectPoses will copy the object when the same name appears
    multiple times (spawning duplicates). Objects with types not present in the scene
    are skipped.
    """
    desired = traj.get('scene', {}).get('object_poses', []) or []
    objectPoses: List[Dict[str, Any]] = []
    skipped_types: Counter = Counter()

    # Keep a round-robin index so we distribute source names if there are several
    rr_idx: Dict[str, int] = {}
    #breakpoint()
    for e in desired:
        #typ = e.get('type') or _parse_legacy_object_id(e.get('objectId', ''))[0]
        typ = e.get('objectName').split("_")[0]
        pos = e.get('position') or (e.get('pose') or {}).get('position')
        rot = e.get('rotation') or (e.get('pose') or {}).get('rotation')
        if not typ or not pos or not rot:
            continue
        nt = _normalize_type_name(typ)
        pool = name_pool.get(nt) or []
        if not pool:
            skipped_types[typ] += 1
            continue
        i = rr_idx.get(nt, 0)
        name = pool[i % len(pool)]
        rr_idx[nt] = i + 1

        objectPoses.append({
            "objectName": name,
            "position": {"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])},
            "rotation": {"x": float(rot.get("x", 0.0)), "y": float(rot.get("y", 0.0)), "z": float(rot.get("z", 0.0))},
        })

    return objectPoses, skipped_types


def _gather_current_movable_payload(event) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Return (payload, type->names) for *all* current pickupable/moveable objects."""
    payload: List[Dict[str, Any]] = []
    type_to_names: Dict[str, List[str]] = {}
    objs = event.metadata.get('objects', []) if event else []
    for o in objs:
        if not (o.get('pickupable') or o.get('moveable')):
            continue
        name = o.get('name'); pos = o.get('position'); rot = o.get('rotation'); typ = o.get('objectType')
        if not (name and pos and rot and typ):
            continue
        # Include ALL movable/pickupable objects to preserve furniture
        # Previously filtered by keywords which removed chairs, sofas, beds, etc.
        payload.append({
            "objectName": name,
            "position": {"x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"])},
            "rotation": {"x": float(rot.get("x", 0.0)), "y": float(rot.get("y", 0.0)), "z": float(rot.get("z", 0.0))},
        })
        nt = _normalize_type_name(typ)
        type_to_names.setdefault(nt, []).append(name)
    #breakpoint()
    return payload, type_to_names


def _overwrite_objects_from_traj(controller: Controller, traj: Dict[str, Any]) -> None:
    """SetObjectPoses with a choice of modes:

    - overwrite: sends only the trajectory-requested items (AI2-THOR removes all unlisted movables)
    - preserve:  starts with all current movables, then *overrides* matching types/names and
                 appends duplicates if trajectory requests more instances than currently exist
    """
    ev = controller.last_event
    name_pool = _object_name_pool_by_type(ev)

    desired, skipped_types = _build_objectposes_from_traj(traj, name_pool)

    baseline, type_to_names = _gather_current_movable_payload(ev)
    if not baseline:
        objectPoses = desired
    else:
        # Create a map of baseline objects by name for quick lookup
        baseline_map = {obj["objectName"]: obj for obj in baseline}
        
        # Track which baseline objects have been updated (to avoid duplicates)
        updated_baseline_names = set()
        new_objects = []
        
        for d in desired:
            obj_name = d["objectName"]
            if obj_name in baseline_map and obj_name not in updated_baseline_names:
                # Update existing object's position and rotation (first occurrence only)
                baseline_map[obj_name]["position"] = d["position"]
                baseline_map[obj_name]["rotation"] = d["rotation"]
                updated_baseline_names.add(obj_name)
            else:
                # Either object doesn't exist in baseline, or already updated
                # Add it to create duplicates if same name appears multiple times
                new_objects.append(d)
        
        # Combine: existing objects (some updated) + new objects (including duplicates)
        objectPoses = list(baseline_map.values()) + new_objects
    
    if not objectPoses:
        print("[OverwriteFromTraj] No object poses could be built; skipping SetObjectPoses")
        return
    #breakpoint()
    controller.step(dict(action='SetObjectPoses', objectPoses=objectPoses))

    if skipped_types:
        print(f"[OverwriteFromTraj] Skipped types with no base objects in this scene: {dict(skipped_types)}")

    # Log diagnostics
    after_objs = controller.last_event.metadata.get('objects', []) if controller.last_event else []
    after_movable = [o for o in after_objs if o.get('pickupable') or o.get('moveable')]
    print(f"[OverwriteFromTraj] After SetObjectPoses: total={len(after_objs)}, movable={len(after_movable)}, specified={len(objectPoses)}")

    return objectPoses
    
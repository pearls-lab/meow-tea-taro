import os
import cv2
import sys
import json
import re
import random
import traceback

import numpy as np
import math
from collections import Counter

from alfred import env
import alfred.gen
import alfworld.gen
import alfred.gen.constants as constants
from alfworld.agents.controller.base import BaseAgent
from alfworld.agents.utils.misc import extract_admissible_commands_with_heuristics


class OracleAgent(BaseAgent):

    def __init__(self, env, traj_data, tw_data, traj_root,
                 load_receps=False, debug=False,
                 goal_desc_human_anns_prob=0.0,
                 use_gt_relations=False,
                 use_explored_receps=False):
        self.openable_points = self.get_openable_points(traj_data)
        # self.explored_receps = tw_data.get('explored_receps', {}) if use_explored_receps else {}
        self.use_gt_relations = use_gt_relations
        self.exploration_frames = []
        super().__init__(env, traj_data, traj_root,
                         load_receps=load_receps, debug=debug,
                         goal_desc_human_anns_prob=goal_desc_human_anns_prob)

    def get_openable_points(self, traj_data):
        scene_num = traj_data['scene']['scene_num']
        openable_json_file = os.path.join(alfworld.gen.__path__[0], 'layouts/FloorPlan%d-openable.json' % scene_num)
        with open(openable_json_file, 'r') as f:
            openable_points = json.load(f)
        return openable_points

    def get_obj_cls_from_metadata(self, name):
        objs = [obj for obj in self.env.last_event.metadata['objects'] if obj['visible'] and name in obj['objectType']]
        return objs[0] if len(objs) > 0 else None

    def get_obj_id_from_metadata(self, object_id):
        objs = [obj for obj in self.env.last_event.metadata['objects'] if object_id == obj['objectId']]
        return objs[0] if len(objs) > 0 else None

    def get_num_interactable_objs(self, recep_id):
        return len([obj for obj in self.env.last_event.metadata['objects'] if obj['visible'] and obj['parentReceptacles'] and recep_id in obj['parentReceptacles']])

    def get_exploration_frames(self):
        return self.exploration_frames

    # Helpers for scene exploration:
    def _get_abs_viewpose(self):
        """Snapshot the agent's exact pose as an absolute TeleportFull action."""
        a = self.env.last_event.metadata['agent']
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

    def get_explored_receps(self):
        return self.receptacles

    # use pre-computed openable points from ALFRED to store receptacle locations
    def explore_scene(self):
        # if self.explored_receps and len(self.explored_receps) > 0:
        #     print("Using pre-computed explored receptacles for ALFWorld Oracle agent...")
        #     return
        agent_height = self.env.last_event.metadata['agent']['position']['y']
        print("Exploring scene for ALFWorld Oracle agent...")
        # teleport to each openable point and record receptacle info
        for openable_object_id, point in self.openable_points.items():
            action = {'action': 'TeleportFull',
                      'x': point[0],
                      'y': agent_height,
                      'z': point[1],
                      'rotateOnTeleport': False,
                      'rotation': point[2],
                      'horizon': point[3]}
            event = self.env.step(action)
            
            for turn in range(2):
                if event.metadata['lastActionSuccess']:
                    new_locs = self._get_abs_viewpose()

                    self.exploration_frames.append(np.array(self.env.last_event.frame[:,:,::-1]))
                    instance_segs = np.array(self.env.last_event.instance_segmentation_frame)
                    color_to_object_id = self.env.last_event.color_to_object_id

                    # find unique instance segs
                    color_count = Counter()
                    for x in range(instance_segs.shape[0]):
                        for y in range(instance_segs.shape[1]):
                            color = instance_segs[y, x]
                            color_count[tuple(color)] += 1

                    for color, num_pixels in color_count.most_common():
                        if color in color_to_object_id:
                            object_id = color_to_object_id[color]
                            object_type = object_id.split('|')[0]
                            if "Basin" in object_id:
                                object_type += "Basin"

                            if object_type in self.STATIC_RECEPTACLES:
                                if object_id not in self.receptacles:
                                    self.receptacles[object_id] = {
                                        'object_id': object_id,
                                        'object_type': object_type,
                                        'locs': new_locs,
                                        'num_pixels': num_pixels,
                                        'num_id': "%s %d" % (object_type.lower(), self.get_next_num_id(object_type, self.receptacles)),
                                        'closed': True if object_type in constants.OPENABLE_CLASS_LIST else None
                                    }
                                elif object_id in self.receptacles and num_pixels > self.receptacles[object_id]['num_pixels']:
                                    self.receptacles[object_id]['locs'] = new_locs
                                    self.receptacles[object_id]['num_pixels'] = num_pixels

                # turn 180 degrees and try again
                event = self.env.step({'action': 'RotateRight'})
                event = self.env.step({'action': 'RotateRight'})


    # ground-truth instance segemetations (with consistent object IDs) from THOR
    def get_instance_seg(self):
        instance_segs = np.array(self.env.last_event.instance_segmentation_frame)
        inst_color_to_object_id = self.env.last_event.color_to_object_id

        # find unique instance segs
        inst_color_count = Counter()
        for x in range(instance_segs.shape[0]):
            for y in range(instance_segs.shape[1]):
                color = instance_segs[y, x]
                inst_color_count[tuple(color)] += 1
        return inst_color_count, inst_color_to_object_id

    # ground-truth object state info maintained by ThorEnv
    def get_object_state(self, object_id):
        is_clean = object_id in self.env.cleaned_objects
        is_hot = object_id in self.env.heated_objects
        is_cool = object_id in self.env.cooled_objects
        is_sliced = 'Sliced' in object_id
        return is_clean, is_cool, is_hot, is_sliced

    def get_admissible_commands(self):
        return extract_admissible_commands_with_heuristics(self.intro, self.frame_desc, self.feedback,
                                                           self.curr_recep, self.inventory)

    def print_frame(self, recep, loc):
        inst_color_count, inst_color_to_object_id = self.get_instance_seg()
        recep_object_id = recep['object_id']

        # for each unique seg add to object dictionary if it's more visible than before
        visible_objects = []
        for color, num_pixels in inst_color_count.most_common():
            if color in inst_color_to_object_id:
                object_id = inst_color_to_object_id[color]
                object_type = object_id.split("|")[0]
                object_metadata = self.get_obj_id_from_metadata(object_id)
                is_obj_in_recep = (object_metadata and object_metadata['parentReceptacles'] and len(object_metadata['parentReceptacles']) > 0 and recep_object_id in object_metadata['parentReceptacles'])
                if object_type in self.OBJECTS and object_metadata and (not self.use_gt_relations or is_obj_in_recep):
                    if object_id not in self.objects:
                        self.objects[object_id] = {
                            'object_id': object_id,
                            'object_type': object_type,
                            'parent': recep['object_id'],
                            'loc': loc,
                            'num_pixels': num_pixels,
                            'num_id': "%s %d" % (object_type.lower() if "Sliced" not in object_id else "sliced-%s" % object_type.lower(),
                                                 self.get_next_num_id(object_type, self.objects))
                        }
                    elif object_id in self.objects and num_pixels > self.objects[object_id]['num_pixels']:
                        self.objects[object_id]['loc'] = loc
                        self.objects[object_id]['num_pixels'] = num_pixels

                    if self.objects[object_id]['num_id'] not in self.inventory:
                        visible_objects.append(self.objects[object_id]['num_id'])

        visible_objects_with_articles = ["a %s," % vo for vo in visible_objects]
        feedback = ""
        if len(visible_objects) > 0:
            feedback = "On the %s, you see %s" % (recep['num_id'], self.fix_and_comma_in_the_end(' '.join(visible_objects_with_articles)))
        elif not recep['closed'] and len(visible_objects) == 0:
            feedback = "On the %s, you see nothing." % (recep['num_id'])

        return visible_objects, feedback

    # Helpers for `step` function:
    def teleport_safely(self):
        """
        Teleports the agent, handling cases where the held object collides with the world.
        Strategy: Send object to void -> Teleport Agent -> Summon Object back.
        """
        
        # 1. Check if we are holding anything
        # In AI2-THOR, metadata['inventoryObjects'] is a list
        inventory = self.env.last_event.metadata['inventoryObjects']
        held_obj_id = inventory[0]['objectId'] if len(inventory) > 0 else None

        # --- CASE A: Hand is Empty ---
        if not held_obj_id:
            return self.navigate(self.curr_loc)

        # --- CASE B: Hand is Full (The Danger Zone) ---
        
        # 1. Try standard teleport first (Optimistic)
        event = self.navigate(self.curr_loc)
        if event.metadata['lastActionSuccess']:
            return event

        # 2. If failed specifically due to collision...
        err = event.metadata['errorMessage']
        if "hand" in err or "collision" in err:
            # print(f"  Collision detected ({err}). Performing Void Maneuver...")

            # A. Send object to the Void (Way below the map)
            # We use TeleportObject to move it out of the way instantly
            event = self.env.step({
                "action": "DropHandObject",
                "forceAction": True 
            })
            if not event.metadata['lastActionSuccess']:
                print(f"error: {event.metadata['errorMessage']}")
            
            # B. Teleport the Agent (Now safe)
            # We pass the original action again
            event = self.navigate(self.curr_loc)
            
            if not event.metadata['lastActionSuccess']:
                print(f"  CRITICAL: Agent teleport failed even without object! {event.metadata['errorMessage']}")
                print(f"error: {event.metadata['errorMessage']}")
                # If this fails, the location itself is bad (inside a wall), not just the hand.
                # You might need to trigger your 'find_valid_interaction_point' again here.

            # C. Summon Object back to Hand
            # Using PickupObject with forceAction=True will teleport it 
            # from the Void directly into the hand, regardless of distance.
            self.env.step({
                "action": "PickupObject",
                "objectId": held_obj_id,
                "forceAction": True
            })
            
        return self.env.last_event

    def find_valid_interaction_point(self, action_str, object_id, receptacle_id=None):
        """
        Manually searches for a valid teleport point where the object is visible.
        Compatible with AI2-THOR 2.1.0 (ALFRED version).
        """
        # 1. Get original state of the object (to revert back later)
        target_obj_id = receptacle_id if action_str == "PutObject" else object_id
        target_obj_meta = None
        for obj in self.env.last_event.metadata['objects']:
            if obj['objectId'] == target_obj_id:
                target_obj_meta = obj
                break
                
        if not target_obj_meta:
            print(f"Check failed: Object {object_id} not found in scene metadata.")
            return None

        original_pos = target_obj_meta['position']
        original_rot = target_obj_meta['rotation']

        # 2. Search Grid (Polar coordinates around target)

        # --- SEARCH PARAMETERS ---
        # 1. Distance: Try closer (0.25) for small items, further (1.5) for big ones
        radii = [0.25, 0.5, 0.75, 1.0, 1.25] 
        
        # 2. Rotation: Standard 8 directions
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        # 3. Horizon (Vertical Look): Look down (30), straight (0), or steep down (60)
        # Many failures happen because the agent isn't looking AT the object
        horizons = [0, -30, 30] 

        print(f"  Starting Omni-Search for {target_obj_id}...")

        for hor in horizons:
            for r in radii:
                for angle_deg in angles:
                    
                    # --- A. Math Setup ---
                    angle_rad = math.radians(angle_deg)
                    nav_target_pos = original_pos 
                    
                    # If Putting, we might need to target the Receptacle's position, not the object's
                    # (Implementation detail: verify logic for receptacle lookup)

                    cand_x = nav_target_pos['x'] + r * math.sin(angle_rad)
                    cand_z = nav_target_pos['z'] + r * math.cos(angle_rad)
                    
                    dx = nav_target_pos['x'] - cand_x
                    dz = nav_target_pos['z'] - cand_z
                    look_yaw = math.degrees(math.atan2(dx, dz))
                    if look_yaw < 0: look_yaw += 360
                    look_yaw = round(look_yaw / 15) * 15 

                    # --- B. Teleport ---
                    # Use the explicit 'standing' and 'horizon' params
                    teleport_action = {
                        "action": "TeleportFull",
                        "x": cand_x,
                        "y": target_obj_meta['position']['y'],
                        "z": cand_z,
                        "rotation": {"x": 0, "y": look_yaw, "z": 0},
                        "horizon": hor,
                        "forceAction": True
                    }
                    
                    event = self.env.step(teleport_action)
                    if not event.metadata['lastActionSuccess']:
                        continue 

                    # --- C. Visibility Check (Optimization) ---
                    # Before trying the physics interaction, verify we can SEE it.
                    # This speeds up the loop significantly.
                    is_visible = False
                    for obj in event.metadata['objects']:
                        if obj['objectId'] == object_id and obj['visible']:
                            is_visible = True
                            break
                    
                    if not is_visible:
                        continue # Don't bother trying to pick up invisible objects

                    # --- D. Interaction Probe ---
                    success = False
                    if action_str == "PickupObject":
                        event = self.env.step({"action": "PickupObject", "objectId": object_id, "forceAction": False})
                        if event.metadata['lastActionSuccess']:
                            success = True
                            # Revert State
                            self.env.step({"action": "TeleportObject", "objectId": object_id, 
                                    "x": original_pos['x'], "y": original_pos['y'], "z": original_pos['z'], 
                                    "rotation": original_rot, "forceAction": True})
                            # Note: You might also need to empty the hand manually via DropHandObject if TeleportObject doesn't clear inventory

                    elif action_str == "PutObject":
                        # (Same Put logic as before)
                        event = self.env.step({"action": "PutObject", "objectId": object_id, "receptacleObjectId": receptacle_id, "forceAction": False})
                        if event.metadata['lastActionSuccess']:
                            success = True
                            self.env.step({"action": "PickupObject", "objectId": object_id, "forceAction": True})

                    elif action_str == "OpenObject":
                        event = self.env.step({"action": "OpenObject", "objectId": object_id, "forceAction": False})
                        if event.metadata['lastActionSuccess']:
                            success = True
                            self.env.step({"action": "CloseObject", "objectId": object_id, "forceAction": True})
                
                    # --- E. Success? ---
                    if success:
                        print(f"  SUCCESS: Found valid point! (R={r}, H={hor}, Angle={angle_deg})")
                        return teleport_action

        return None # Failed

    def step(self, action_str):
        event = None
        self.feedback = "Nothing happens."

        try:
            cmd = self.parse_command(action_str) 

            if cmd['action'] == self.Action.GOTO:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                if recep and recep['num_id'] == self.curr_recep:
                    return self.feedback
                self.curr_loc = recep['locs']
                # event = self.navigate(self.curr_loc)
                event = self.teleport_safely()
                self.curr_recep = recep['num_id']
                self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)

                # feedback conditions
                loc_id = list(self.receptacles.keys()).index(recep['object_id'])
                loc_feedback = "You arrive at loc %s. " % loc_id
                state_feedback = "The {} is {}. ".format(self.curr_recep, "closed" if recep['closed'] else "open") if recep['closed'] is not None else ""
                loc_state_feedback = loc_feedback + state_feedback
                self.feedback = loc_state_feedback + self.feedback if "closed" not in state_feedback else loc_state_feedback
                self.frame_desc = str(self.feedback)

            elif cmd['action'] == self.Action.PICK:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                print(f"Visible objects: {self.visible_objects}")
                if obj not in self.visible_objects:
                    for visible_obj in self.visible_objects:
                        if visible_obj.split()[0] == obj.split()[0]:
                            obj = visible_obj
                            with open("/root/data/alfworld/sft_info/pick_debug.txt", "a") as f:
                                f.write(f"Change {cmd['obj']} to {obj}\n")
                            break

                if obj in self.visible_objects:
                    object = self.get_object(obj, self.objects)
                    # (Optional) Close object if it's openable (needed for toggleable objects)
                    event = self.env.step({'action': "CloseObject",
                                            'objectId': object['object_id'],
                                            'forceAction': True})
                    if object['object_id'] in self.receptacles:
                        self.receptacles[object['object_id']]['closed'] = True
                    # Initial attempt to pick up the object
                    event = self.env.step({'action': "PickupObject",
                                           'objectId': object['object_id'],
                                           'forceAction': True})
                    # If initial pickup fails, try finding the valid interaction points
                    if not event.metadata['lastActionSuccess']:
                        if self.debug:
                            print(f"Initial PickupObject failed: {event.metadata['errorMessage']}")
                            print("Finding interactable positions for picking up the object...")

                        teleport_action = self.find_valid_interaction_point("PickupObject", object_id=object['object_id'], receptacle_id=None)
                        if teleport_action is not None:
                            event = self.env.step(teleport_action)
                            event = self.env.step({'action': "PickupObject",
                                                   'objectId': object['object_id'],
                                                   'forceAction': True})

                    if event.metadata['lastActionSuccess']:
                        self.inventory.append(object['num_id'])
                        self.feedback = "You pick up the %s from the %s." % (obj, tar)
                    else:
                        print("Pickup failed: ", event.metadata['errorMessage'])


            elif cmd['action'] == self.Action.PUT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                recep = self.get_object(tar, self.receptacles)

                # Initial attempt to place the object
                event = self.env.step({
                    'action': "PutObject",
                    'objectId': self.env.last_event.metadata['inventoryObjects'][0]['objectId'],
                    'receptacleObjectId': recep['object_id'],
                    'forceAction': True,
                    'placeStationary': True,
                })
                # If initial put object fails, try finding the valid interaction points
                if not event.metadata['lastActionSuccess']:
                    if self.debug:
                        print(f"Initial PutObject failed: {event.metadata['errorMessage']}")
                        print("Finding interactable positions for better placement...")

                    teleport_action = self.find_valid_interaction_point("PutObject", object_id=None, receptacle_id=recep['object_id'])
                    if teleport_action is not None:
                        event = self.env.step(teleport_action)
                        event = self.env.step({
                            'action': "PutObject",
                                'objectId': self.env.last_event.metadata['inventoryObjects'][0]['objectId'],
                                'receptacleObjectId': recep['object_id'],
                                'forceAction': True,
                                'placeStationary': True,
                            })
                
                if event.metadata['lastActionSuccess']:
                    self.inventory.pop()
                    self.feedback = "You put the %s %s the %s." % (obj, rel, tar)
                else:
                    print("PutObject failed: ", event.metadata['errorMessage'])

            elif cmd['action'] == self.Action.OPEN:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                event = self.env.step({'action': "OpenObject",
                                       'objectId': recep['object_id'],
                                       'forceAction': True})
                
                # If initial put object fails, try finding the valid interaction points
                if not event.metadata['lastActionSuccess']:
                    if self.debug:
                        print(f"Initial OpenObject failed: {event.metadata['errorMessage']}")
                        print("Finding interactable positions for better placement...")

                    teleport_action = self.find_valid_interaction_point("OpenObject", object_id=recep['object_id'])
                    if teleport_action is not None:
                        event = self.env.step(teleport_action)
                        event = self.env.step({'action': "OpenObject",
                                               'objectId': recep['object_id'],
                                               'forceAction': True})
                        
                self.receptacles[recep['object_id']]['closed'] = False
                self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)
                action_feedback = "You open the %s. The %s is open. " % (target, target)
                self.feedback = action_feedback + self.feedback.replace("On the %s" % target, "In it")
                self.frame_desc = str(self.feedback)

            elif cmd['action'] == self.Action.CLOSE:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                event = self.env.step({'action': "CloseObject",
                                       'objectId': recep['object_id'],
                                       'forceAction': True})
                self.receptacles[recep['object_id']]['closed'] = True
                self.feedback = "You close the %s." % target

            elif cmd['action'] == self.Action.TOGGLE:
                target = cmd['tar']
                obj = self.get_object(target, self.objects)
                event = self.env.step({'action': "ToggleObjectOn",
                                       'objectId': obj['object_id'],
                                       'forceAction': True})
                self.feedback = "You turn on the %s." % target

            elif cmd['action'] == self.Action.HEAT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
                recep = self.get_object(tar, self.receptacles)

                # open the microwave, heat the object, take the object, close the microwave
                events = []
                events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PutObject', 'objectId': obj_id, 'receptacleObjectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'Pass'}))
                events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PickupObject', 'objectId': obj_id, 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You heat the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.CLEAN:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.env.last_event.metadata['inventoryObjects'][0]
                sink = self.get_obj_cls_from_metadata('BathtubBasin' if "bathtubbasin" in tar else "SinkBasin")
                faucet = self.get_obj_cls_from_metadata('Faucet')

                # put the object in the sink, turn on the faucet, turn off the faucet, pickup the object
                events = []
                events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': sink['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': faucet['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'Pass'}))
                events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': faucet['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You clean the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.COOL:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.env.last_event.metadata['inventoryObjects'][0]
                fridge = self.get_obj_cls_from_metadata('Fridge')

                # open the fridge, put the object inside, close the fridge, open the fridge, pickup the object
                events = []
                events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'Pass'}))
                events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))
                events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.feedback = "You cool the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.SLICE:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                object = self.get_object(obj, self.objects)
                inventory_objects = self.env.last_event.metadata['inventoryObjects']
                if 'Knife' in inventory_objects[0]['objectType']:
                    event = self.env.step({'action': "SliceObject",
                                           'objectId': object['object_id']})
                self.feedback = "You slice %s with the %s" % (obj, tar)

            elif cmd['action'] == self.Action.INVENTORY:
                if len(self.inventory) > 0:
                    self.feedback = "You are carrying: a %s" % (self.inventory[0])
                else:
                    self.feedback = "You are not carrying anything."

            elif cmd['action'] == self.Action.EXAMINE:
                target = cmd['tar']
                receptacle = self.get_object(target, self.receptacles)
                object = self.get_object(target, self.objects)

                if receptacle:
                    self.visible_objects, self.feedback = self.print_frame(receptacle, self.curr_loc)
                    self.frame_desc = str(self.feedback)
                elif object:
                    self.feedback = self.print_object(object)

            elif cmd['action'] == self.Action.LOOK:
                if self.curr_recep == "nothing":
                    self.feedback = "You are in the middle of a room. Looking quickly around you, you see nothing."
                else:
                    self.feedback = "You are facing the %s. Next to it, you see nothing." % self.curr_recep

        except:
            if self.debug:
                print(traceback.format_exc())

        if event and not event.metadata['lastActionSuccess']:
            self.feedback = "Nothing happens."
            if self.debug:
                print(event.metadata['errorMessage'])

        return self.feedback


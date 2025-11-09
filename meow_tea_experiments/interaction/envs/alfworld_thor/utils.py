def find_correct_object(traj_object_id, event_metadata):
    """
    Find the correct object from the scene metadata.

    Args:
        traj_object_id: The objectId from trajectory (may be outdated)
        event_metadata: Metadata from env.last_event.metadata
    
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


def fix_object_ids_in_action(env, traj_api_cmd, event_metadata):
    """
    Fix objectIds in the action command if they don't exist in the scene.
    
    Args:
        traj_api_cmd: The API action command from trajectory
        event_metadata: Metadata from env.last_event.metadata
    
    Returns:
        Fixed action command with correct objectIds
    """
    fixed_cmd = traj_api_cmd.copy()
    
    # Check for objectId
    if 'objectId' in fixed_cmd:
        #breakpoint()
        traj_obj_id = fixed_cmd['objectId']
        correct_obj = find_correct_object(traj_obj_id, event_metadata)
        if correct_obj:
            if correct_obj['objectId'] != traj_obj_id:
                print(f"  [Fix] objectId: {traj_obj_id} -> {correct_obj['objectId']}")
            fixed_cmd['objectId'] = correct_obj['objectId']

            if traj_api_cmd.get('action') in ['ToggleObjectOn', 'ToggleObjectOff']:
                # Update object states in env
                print(f"  [Info] Updating object state for: {correct_obj['objectId']}")
                env.step(dict(action="ToggleObjectOff", objectId=correct_obj["objectId"], forceAction=True))
                # env.step(dict(action='SetObjectStates',
                        #SetObjectStates={'objectType': o['objectType'], 'stateChange': 'toggleable', 'isToggled': False})) # 'stateChange': 'toggleable' doesn't work
                        # SetObjectStates={'objectType': correct_obj['objectType'], 'isToggled': False}))
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
    
    return fixed_cmd

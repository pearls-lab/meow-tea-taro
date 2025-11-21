"""
Extract trajectory data from pick_and_place_simple folders.
Organize and rename them with task metadata.
"""

import json
import os
from tqdm import tqdm


def extract_pick_and_place_simple_trajectories(
    source_root: str = "/root/data/alfworld/raw/valid_seen",
    output_dir: str = "/root/data/alfworld/pick_and_place_simple_extracted"
):
    """
    Extract all traj_data.json files from pick_and_place_simple folders in valid_seen.

    Args:
        source_root: Root directory to search for pick_and_place_simple folders (default: valid_seen)
        output_dir: Directory to save extracted trajectory data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all traj_data.json files under pick_and_place_simple folders
    traj_files = []
    for root, _, files in os.walk(source_root):
        if "pick_and_place_simple" in root and "traj_data.json" in files:
            traj_files.append(os.path.join(root, "traj_data.json"))

    print(f"Found {len(traj_files)} trajectory files under pick_and_place_simple folders")

    # Process each trajectory file
    metadata_list = []
    task_counter = 1

    for traj_file_path in tqdm(traj_files, desc="Processing trajectories"):
        try:
            # Load the trajectory data
            with open(traj_file_path, 'r') as f:
                traj_data = json.load(f)

            # Extract metadata from the trajectory
            task_desc = traj_data.get('turk_annotations', {}).get('anns', [{}])[0].get('task_desc', 'Unknown task')

            # Get the folder structure info
            relative_path = os.path.relpath(traj_file_path, source_root)
            path_parts = relative_path.split(os.sep)

            # Find the pick_and_place_simple folder name
            pick_and_place_folder = None
            for part in path_parts:
                if part.startswith("pick_and_place_simple"):
                    pick_and_place_folder = part
                    break

            # Construct task_dir (the nested folder structure from pick_and_place_simple onwards)
            if pick_and_place_folder:
                folder_idx = path_parts.index(pick_and_place_folder)
                task_dir = os.path.join(*path_parts[folder_idx:-1])  # Exclude traj_data.json
            else:
                task_dir = "unknown"

            # Create output filename
            output_filename = f"task1_{task_counter}.json"
            output_filepath = os.path.join(output_dir, output_filename)

            # Create metadata entry
            metadata = {
                "task_type": "task1",
                "task_dir": task_dir,
                "task_id": task_counter,
                "task_desc": task_desc,
                "source_path": relative_path
            }

            # Save the trajectory data
            with open(output_filepath, 'w') as f:
                json.dump(traj_data, f, indent=2)

            # Add to metadata list
            metadata_list.append(metadata)
            task_counter += 1

        except Exception as e:
            print(f"Error processing {traj_file_path}: {e}")

    # Save metadata index file
    metadata_output_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\n✓ Extracted {len(metadata_list)} trajectories")
    print(f"✓ Saved to: {output_dir}")
    print(f"✓ Metadata saved to: {metadata_output_path}")

    # Print sample metadata
    if metadata_list:
        print("\nSample metadata entries:")
        for meta in metadata_list[:3]:
            print(f"  {meta['task_id']}: {meta['task_desc']}")
        if len(metadata_list) > 3:
            print(f"  ... and {len(metadata_list) - 3} more")

    return metadata_list




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract trajectory data from pick_and_place_simple folders in valid_seen"
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="/root/data/alfworld/raw/valid_seen",
        help="Root directory to search for pick_and_place_simple folders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/data/alfworld/pick_and_place_simple_extracted",
        help="Directory to save extracted trajectory data"
    )

    args = parser.parse_args()

    extract_pick_and_place_simple_trajectories(
        source_root=args.source_root,
        output_dir=args.output_dir
    )

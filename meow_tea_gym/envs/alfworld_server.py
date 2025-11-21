"""
ALFWorld Environment Server (Python 3.6+)
Manages the ALFWorld environment and communicates with vLLM client via sockets.
"""

import json
import yaml
import os
import socket
import threading
from PIL import Image
import io
import base64
import argparse
import sys
from alfred.env.thor_env import ThorEnv
from alfworld.agents.controller import OracleAgent

# Configuration
with open("alfworld/config.yaml") as reader:
    config = yaml.safe_load(reader)

class MockArgument:
    """Fake argument class to reuse exisiting functions"""
    reward_config = 'alfworld/rewards.json'


# Default paths (can be overridden by command-line arguments)
DATA_DIR = "/root/data/alfworld/valid_seen_task1/"
MAX_TURNS = 50


def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes, matching training data format."""
    img_byte_arr = io.BytesIO()
    # Convert to RGB to match training data processing
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 for JSON transmission."""
    return base64.b64encode(image_bytes).decode('utf-8')


class ALFWorldServer:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.env = None
        self.controller = None
        self.event = None
        self.task_desc = None
        self.messages = []
        self.images = []
        self.trajectory = []
        self.turn_count = 0

    def start(self):
        """Start the server and listen for client connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[SERVER] ALFWorld server listening on {self.host}:{self.port}")

        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"[SERVER] Client connected from {addr}")
            self.handle_client()
        finally:
            self.cleanup()

    def handle_client(self):
        """Handle client requests in a loop."""
        while True:
            try:
                # Receive message from client (increased buffer for larger requests)
                data = self.client_socket.recv(65536).decode('utf-8')
                if not data:
                    break

                request = json.loads(data)
                response = self.process_request(request)

                # Send response back to client (ensure complete message is sent)
                response_data = json.dumps(response).encode('utf-8')
                self.client_socket.sendall(response_data)
            except json.JSONDecodeError as e:
                print(f"[SERVER] JSON decode error: {e}")
                break
            except Exception as e:
                print(f"[SERVER] Error: {e}")
                break

    def process_request(self, request):
        """Process different types of requests from the client."""
        command = request.get('command')

        if command == 'init':
            return self.cmd_init(request)
        elif command == 'step':
            return self.cmd_step(request)
        elif command == 'reset_task':
            return self.cmd_reset_task(request)
        elif command == 'get_state':
            return self.cmd_get_state(request)
        elif command == 'is_goal_satisfied':
            return self.cmd_is_goal_satisfied(request)
        elif command == 'shutdown':
            return self.cmd_shutdown(request)
        else:
            return {'success': False, 'error': f'Unknown command: {command}'}

    def cmd_init(self, request):
        """Initialize a new task."""
        traj_file = request.get('traj_file')

        try:
            traj_root = os.path.dirname(traj_file)
            with open(os.path.join(DATA_DIR, traj_file)) as f:
                traj_data = json.load(f)

            # Reset environment
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']
            scene_name = 'FloorPlan%d' % scene_num

            if self.env is None:
                self.env = ThorEnv(x_display='0')
            
            self.env.reset(scene_name)

            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            self.event = self.env.step(dict(traj_data['scene']['init_action']))

            # Save initial state before OracleAgent explores
            initial_agent_state = {
                'position': self.env.last_event.metadata['agent']['position'].copy(),
                'rotation': self.env.last_event.metadata['agent']['rotation'].copy(),
                'horizon': self.env.last_event.metadata['agent']['cameraHorizon']
            }

            controller_type = config['controller']['type']
            goal_desc_human_anns_prob = config['env']['goal_desc_human_anns_prob']
            load_receps =   config['controller']['load_receps']
            # debug = config['controller']['debug']
            debug = True

            controller = OracleAgent(self.env, traj_data, None, traj_root,
                                    load_receps=load_receps, debug=debug,
                                    goal_desc_human_anns_prob=goal_desc_human_anns_prob)
            print(f"Initial state: {controller.intro}")

            # Restore agent to exact initial position
            self.env.step({
                'action': 'TeleportFull',
                'x': initial_agent_state['position']['x'],
                'y': initial_agent_state['position']['y'],
                'z': initial_agent_state['position']['z'],
                'rotateOnTeleport': False,
                'rotation': initial_agent_state['rotation'],
                'horizon': initial_agent_state['horizon'],
            })

            self.event = self.env.step(dict(traj_data['scene']['init_action']))

            args = MockArgument()
            self.env.set_task(traj_data, args, reward_type='dense') 

            self.task_desc = traj_data['turk_annotations']['anns'][0]['task_desc']
            print(f"[SERVER] Task initialized: {self.task_desc}")
            self.controller = controller

            # Initialize conversation
            self.messages = []
            self.images = []
            self.trajectory = [self.controller.intro]
            self.turn_count = 0

            # Get initial image
            curr_img = self.event.frame if isinstance(self.event.frame, Image.Image) else Image.fromarray(self.event.frame)
            self.images.append(image_to_bytes(curr_img))

            return {
                'success': True,
                'task_desc': self.task_desc,
                'intro': self.controller.intro,
                'initial_image': bytes_to_base64(self.images[0])
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[SERVER] Init error: {error_msg}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': error_msg}

    def cmd_step(self, request):
        """Execute an action in the environment."""
        action_str = request.get('action')

        try:
            # Execute action
            feedback = self.controller.step(action_str)
            self.event = self.env.last_event

            # Update trajectory
            self.trajectory.append(action_str)
            self.trajectory.append(feedback)
            self.turn_count += 1

            # Get current image
            curr_img = self.event.frame if isinstance(self.event.frame, Image.Image) else Image.fromarray(self.event.frame)
            current_image_bytes = image_to_bytes(curr_img)

            # Store image for next turn (as bytes, matching training data format)
            self.images.append(current_image_bytes)

            response = {
                'success': True,
                'action': action_str,
                'feedback': feedback,
                'current_image': bytes_to_base64(current_image_bytes),
                'action_success': self.event.metadata['lastActionSuccess'],
                'error_message': self.event.metadata.get('errorMessage', ''),
                'goal_satisfied': self.env.get_goal_satisfied(),
                'turn_count': self.turn_count
            }
            return response
        except Exception as e:
            error_msg = str(e)
            print(f"[SERVER] Step error: {error_msg}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': error_msg}

    def cmd_reset_task(self, request):
        """Reset for a new task."""
        self.messages = []
        self.images = []
        self.trajectory = []
        self.turn_count = 0
        return {'success': True}

    def cmd_get_state(self, request):
        """Get current state information."""
        return {
            'success': True,
            'turn_count': self.turn_count,
            'trajectory': self.trajectory,
            'images_count': len(self.images)
        }

    def cmd_is_goal_satisfied(self, request):
        """Check if goal is satisfied."""
        try:
            goal_satisfied = self.env.get_goal_satisfied()
            return {'success': True, 'goal_satisfied': goal_satisfied}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def cmd_shutdown(self, request):
        """Shutdown the server."""
        return {'success': True, 'message': 'Server shutting down'}

    def cleanup(self):
        """Clean up resources."""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        if self.env:
            self.env.stop()
        print("[SERVER] Server shutdown complete")


def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description='ALFWorld Environment Server')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--data-dir', type=str, default='/root/data/alfworld/valid_seen_task1/',
                        help='Directory containing task trajectory files')

    args = parser.parse_args()

    # Update global DATA_DIR if specified
    DATA_DIR = args.data_dir

    server = ALFWorldServer(host=args.host, port=args.port)
    server.start()


if __name__ == '__main__':
    main()

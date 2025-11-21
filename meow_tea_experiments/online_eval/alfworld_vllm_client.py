"""
vLLM Client for ALFWorld (Python 3.11+)
Handles model inference and communicates with ALFWorld server via sockets.
"""

import json
import os
import socket
import time
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor


def base64_to_image(image_b64: str) -> Image.Image:
    """Convert base64 string to PIL Image, matching training data format."""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    return image


class ALFWorldClient:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.llm = None
        self.processor = None

    def connect(self, max_retries=10):
        """Connect to the ALFWorld server."""
        for attempt in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host, self.port))
                self.connected = True
                print(f"[CLIENT] Connected to server at {self.host}:{self.port}")
                return True
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    print(f"[CLIENT] Connection attempt {attempt + 1}/{max_retries} failed, retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"[CLIENT] Failed to connect after {max_retries} attempts")
                    return False

    def send_request(self, request):
        """Send a request to the server and get response."""
        if not self.connected:
            raise RuntimeError("Not connected to server")

        try:
            # Send request
            request_data = json.dumps(request).encode('utf-8')
            self.socket.send(request_data)

            # Receive response (increased buffer size for image data)
            # Images can be large when base64 encoded
            response_data = b''
            while True:
                chunk = self.socket.recv(65536)  # 64KB buffer
                if not chunk:
                    break
                response_data += chunk
                # Try to decode - if we have a complete JSON message, return it
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    return response
                except json.JSONDecodeError:
                    # Not complete yet, keep receiving
                    continue

            # If we get here, connection was closed without data
            if not response_data:
                raise RuntimeError("Server closed connection without sending response")

            return json.loads(response_data.decode('utf-8'))
        except Exception as e:
            print(f"[CLIENT] Error in communication: {e}")
            raise

    def init_task(self, traj_file):
        """Initialize a new task on the server."""
        request = {
            'command': 'init',
            'traj_file': traj_file
        }
        response = self.send_request(request)
        if not response.get('success'):
            raise RuntimeError(f"Failed to init task: {response.get('error')}")
        return response

    def step(self, action):
        """Execute an action on the server."""
        request = {
            'command': 'step',
            'action': action
        }
        response = self.send_request(request)
        if not response.get('success'):
            raise RuntimeError(f"Step failed: {response.get('error')}")
        return response

    def reset_task(self):
        """Reset task state."""
        request = {'command': 'reset_task'}
        response = self.send_request(request)
        return response

    def is_goal_satisfied(self):
        """Check if goal is satisfied."""
        request = {'command': 'is_goal_satisfied'}
        response = self.send_request(request)
        if not response.get('success'):
            raise RuntimeError(f"Error checking goal: {response.get('error')}")
        return response.get('goal_satisfied', False)

    def get_state(self):
        """Get current state."""
        request = {'command': 'get_state'}
        response = self.send_request(request)
        return response

    def shutdown(self):
        """Shutdown the server."""
        request = {'command': 'shutdown'}
        try:
            self.send_request(request)
        except:
            pass
        self.disconnect()

    def disconnect(self):
        """Disconnect from server."""
        if self.socket:
            self.socket.close()
            self.connected = False
        print("[CLIENT] Disconnected from server")

    def load_model(self, model_path, gpu_memory_utilization=0.7, device_ids=None, max_model_len=None):
        """Load vLLM model and processor.

        Args:
            model_path: Path to the model
            gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0), default 0.7
            device_ids: List of GPU IDs to use, e.g., [0] or [0, 1]. If None, uses all available GPUs.
            max_model_len: Maximum model length for KV cache. If None, uses model default.
                          Reduce this if you get "KV cache is needed" errors.
                          Example: 4096, 8192, 16384
        """
        print(f"[CLIENT] Loading model from {model_path}")
        print(f"[CLIENT] GPU memory utilization: {gpu_memory_utilization}")
        if device_ids:
            print(f"[CLIENT] Using GPUs: {device_ids}")
        if max_model_len:
            print(f"[CLIENT] Max model length: {max_model_len}")

        kwargs = {
            'model': model_path,
            'trust_remote_code': True,
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': gpu_memory_utilization,
        }

        # Add max_model_len if specified
        if max_model_len:
            kwargs['max_model_len'] = max_model_len

        # If device_ids specified, set CUDA_VISIBLE_DEVICES
        if device_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

        self.llm = LLM(**kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("[CLIENT] Model loaded successfully")

    def run_inference(self, traj_file, max_turns=50):
        """Run inference for a single trajectory.

        Returns:
            tuple: (trajectory, goal_satisfied) - the trajectory and whether goal was satisfied
        """
        print(f"\n[CLIENT] Starting inference for {traj_file}")

        # Initialize task on server
        format_prompt = "\n\nAlways respond with a command starting with a verb using the following format:\bgo to <location>\btake <object> from <receptacle>\bopen <object>\bclose <object>\bput <object> in <receptacle>\blook\n\n"
        init_response = self.init_task(traj_file)
        task_desc = init_response.get('task_desc')
        intro = init_response.get('intro')
        print(f"[CLIENT] Task: {task_desc}")

        # Initialize conversation
        messages = []
        trajectory = [intro + format_prompt]
        goal_satisfied_final = False  # Track final goal satisfaction

        # First turn setup
        user_format = "\ncurrent state: {text_obs}\nstate shown in image: <|vision_start|><|image_pad|><|vision_end|>\n\nyour action: "
        messages.append({"role": "user", "content": user_format.format(text_obs=intro + format_prompt)})

        # Convert base64 images to PIL Images (matching training data format)
        images = [base64_to_image(init_response.get('initial_image'))]

        # Main loop
        for t in range(max_turns):
            print(f"\n[CLIENT] === Turn {t + 1} ===")

            # Prepare model input
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Get model's action (vLLM expects PIL Images)
            image_data = {"image": images} if images else None

            sampling_params = SamplingParams(
                temperature=0.7,
                max_tokens=50,
                stop=["\n", self.processor.tokenizer.eos_token]
            )

            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": image_data,
                },
                sampling_params=sampling_params
            )
            action_str = outputs[0].outputs[0].text.strip()
            print(f"[CLIENT] Model Action: {action_str}")

            # Execute action on server
            step_response = self.step(action_str)

            feedback = step_response.get('feedback')
            current_image = step_response.get('current_image')
            action_success = step_response.get('action_success')
            error_message = step_response.get('error_message', '')
            goal_satisfied = step_response.get('goal_satisfied')

            # Update trajectory and conversation
            trajectory.append(action_str)
            trajectory.append(feedback)
            messages.append({"role": "assistant", "content": action_str})

            print(f"[CLIENT] Feedback: {feedback}")
            if not action_success:
                print(f"[CLIENT] Action failed: {error_message}")

            # Check completion
            if goal_satisfied:
                print("[CLIENT] Goal satisfied!")
                goal_satisfied_final = True
                break

            # Prepare for next turn
            # Convert base64 image to PIL Image (matching training data format)
            images.append(base64_to_image(current_image))
            messages.append({"role": "user", "content": user_format.format(text_obs=trajectory[t * 2])})

        print(f"[CLIENT] Inference completed after {t + 1} turns")
        self.reset_task()
        return trajectory, goal_satisfied_final

    def run_batch_inference(self, data_dir, output_dir, max_turns=50):
        """Run inference on a batch of tasks."""
        os.makedirs(output_dir, exist_ok=True)

        file_list = sorted(os.listdir(data_dir))
        goals_satisfied = 0
        goals_total = 0

        for traj_file in tqdm(file_list, desc="Processing tasks"):
            if not traj_file.endswith('.json'):
                continue

            try:
                trajectory, goal_satisfied = self.run_inference(traj_file, max_turns=max_turns)

                # Save trajectory
                out_file = os.path.join(output_dir, traj_file)
                with open(out_file, 'w') as f:
                    json.dump(trajectory, f, indent=4)

                # Track goal satisfaction
                if goal_satisfied:
                    goals_satisfied += 1
                goals_total += 1

                print(f"[CLIENT] Saved trajectory to {out_file} - Goal: {'✓' if goal_satisfied else '✗'}")
            except Exception as e:
                print(f"[CLIENT] Exception for {traj_file}: {e}")
                goals_total += 1

        # Calculate and print completion rate
        if goals_total > 0:
            completion_rate = (goals_satisfied / goals_total) * 100
            print(f"\n[CLIENT] =" * 50)
            print(f"[CLIENT] GOAL COMPLETION RATE: {goals_satisfied}/{goals_total} ({completion_rate:.2f}%)")
            print(f"[CLIENT] =" * 50)

            # Save stats to file
            stats = {
                'goals_satisfied': goals_satisfied,
                'goals_total': goals_total,
                'completion_rate': completion_rate
            }
            stats_file = os.path.join(output_dir, 'completion_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"[CLIENT] Stats saved to: {stats_file}")
        else:
            print("[CLIENT] No tasks processed")

        print("[CLIENT] Batch inference complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='vLLM Client for ALFWorld')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--model-path', type=str, default='/root/checkpoints/global_step_795/',
                        help='Path to vLLM model')
    parser.add_argument('--data-dir', type=str, default='/root/data/alfworld/valid_seen_task1/',
                        help='Directory with task data')
    parser.add_argument('--output-dir', type=str, default='/root/data/eval/',
                        help='Output directory for results')
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=None,
                        help='GPU IDs to use (e.g., --gpu-ids 0 1). If not specified, uses all available GPUs.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.7,
                        help='GPU memory utilization ratio (0.0-1.0, default: 0.7)')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Maximum model length for KV cache (e.g., 4096, 8192, 16384). '
                             'Reduce this if you get "KV cache is needed" errors.')

    args = parser.parse_args()

    # Load model
    client = ALFWorldClient(host=args.host, port=args.port)
    client.load_model(args.model_path,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      device_ids=args.gpu_ids,
                      max_model_len=args.max_model_len)

    # Connect to server
    if not client.connect():
        print("[CLIENT] Failed to connect to server. Make sure the server is running.")
        return

    try:
        # Run batch inference
        client.run_batch_inference(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_turns=25
        )
    finally:
        client.shutdown()


if __name__ == '__main__':
    main()

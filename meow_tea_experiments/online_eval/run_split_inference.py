"""
Utility script to manage and run both ALFWorld server and vLLM client processes.
Handles starting, monitoring, and stopping both processes.
"""

import subprocess
import time
import os
import signal
import argparse
import sys


class ProcessManager:
    def __init__(self, server_host='localhost', server_port=5555):
        self.server_host = server_host
        self.server_port = server_port
        self.server_process = None
        self.client_process = None

    def start_server(self, python_executable='python'):
        """Start the ALFWorld server in a subprocess."""
        print(f"[MANAGER] Starting ALFWorld server...")
        try:
            self.server_process = subprocess.Popen(
                [python_executable, 'alfworld_server.py',
                 '--host', self.server_host,
                 '--port', str(self.server_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"[MANAGER] Server started with PID {self.server_process.pid}")
            # Wait for server to start
            time.sleep(3)
            return True
        except Exception as e:
            print(f"[MANAGER] Failed to start server: {e}")
            return False

    def start_client(self, client_args, python_executable='python'):
        """Start the vLLM client in a subprocess."""
        print(f"[MANAGER] Starting vLLM client...")
        try:
            cmd = [python_executable, 'alfworld_vllm_client.py',
                   '--host', self.server_host,
                   '--port', str(self.server_port)]
            cmd.extend(client_args)

            self.client_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"[MANAGER] Client started with PID {self.client_process.pid}")
            return True
        except Exception as e:
            print(f"[MANAGER] Failed to start client: {e}")
            return False

    def monitor_processes(self):
        """Monitor both processes and print their output."""
        while True:
            # Check server
            if self.server_process:
                if self.server_process.poll() is not None:
                    print("[MANAGER] Server process exited")
                    stdout, stderr = self.server_process.communicate()
                    if stdout:
                        print("[SERVER STDOUT]", stdout)
                    if stderr:
                        print("[SERVER STDERR]", stderr)
                    break

            # Check client
            if self.client_process:
                if self.client_process.poll() is not None:
                    print("[MANAGER] Client process exited")
                    stdout, stderr = self.client_process.communicate()
                    if stdout:
                        print("[CLIENT STDOUT]", stdout)
                    if stderr:
                        print("[CLIENT STDERR]", stderr)
                    break

            time.sleep(1)

    def stop_processes(self):
        """Stop both processes gracefully."""
        print("[MANAGER] Stopping processes...")

        if self.client_process:
            try:
                self.client_process.terminate()
                self.client_process.wait(timeout=5)
                print("[MANAGER] Client process stopped")
            except subprocess.TimeoutExpired:
                self.client_process.kill()
                print("[MANAGER] Client process killed")

        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                print("[MANAGER] Server process stopped")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                print("[MANAGER] Server process killed")

    def run(self, client_args, server_python='python', client_python='python'):
        """Run both processes."""
        try:
            # Start server (Python 3.6)
            if not self.start_server(python_executable=server_python):
                print("[MANAGER] Failed to start server")
                return False

            # Start client (Python 3.11)
            if not self.start_client(client_args, python_executable=client_python):
                print("[MANAGER] Failed to start client")
                self.stop_processes()
                return False

            # Monitor processes
            self.monitor_processes()

            return True
        except KeyboardInterrupt:
            print("\n[MANAGER] Interrupted by user")
            self.stop_processes()
        except Exception as e:
            print(f"[MANAGER] Error: {e}")
            self.stop_processes()
            return False

    def __del__(self):
        """Ensure processes are cleaned up."""
        self.stop_processes()


def main():
    parser = argparse.ArgumentParser(
        description='Manage ALFWorld server and vLLM client processes'
    )
    parser.add_argument('--host', type=str, default='localhost',
                        help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5555,
                        help='Server port (default: 5555)')
    parser.add_argument('--server-python', type=str, default='python',
                        help='Python executable for server (Python 3.6+)')
    parser.add_argument('--client-python', type=str, default='python',
                        help='Python executable for client (Python 3.11+)')
    parser.add_argument('--model-path', type=str, default='/root/checkpoints/global_step_795/',
                        help='Path to vLLM model')
    parser.add_argument('--data-dir', type=str, default='/root/data/alfworld/valid_seen/',
                        help='Directory with task data')
    parser.add_argument('--traj-dir', type=str, default='/root/data/alfworld/valid_seen_v5_high_trajs/',
                        help='Directory with trajectory files')
    parser.add_argument('--output-dir', type=str, default='/root/data/alfworld/valid_seen_v5_vlm_inference_trajs/',
                        help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process')

    args = parser.parse_args()

    # Build client arguments
    client_args = [
        '--model-path', args.model_path,
        '--data-dir', args.data_dir,
        '--traj-dir', args.traj_dir,
        '--output-dir', args.output_dir,
    ]
    if args.num_samples:
        client_args.extend(['--num-samples', str(args.num_samples)])

    # Create and run manager
    manager = ProcessManager(server_host=args.host, server_port=args.port)
    success = manager.run(
        client_args=client_args,
        server_python=args.server_python,
        client_python=args.client_python
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

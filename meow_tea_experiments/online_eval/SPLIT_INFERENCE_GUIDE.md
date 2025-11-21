# Split vLLM and ALFWorld Inference

This guide explains how to run vLLM inference and ALFWorld environment in separate processes with different Python versions to handle compatibility issues.

## Problem

The original `alfworld_mm.py` loads both:
- **vLLM** (requires Python 3.11+)
- **ALFWorld** (requires Python 3.6+)

These have conflicting Python version requirements and cannot run in the same process.

## Solution

Split the inference pipeline into two separate processes that communicate via **socket-based IPC (Inter-Process Communication)**:

### Architecture

```
┌──────────────────────────────┐
│  vLLM Client (Python 3.11)   │
│  - Load vLLM model           │
│  - Generate actions          │
│  - Process observations      │
└──────────┬──────────────────┘
           │ JSON over TCP
           │ Port 5555
           ↓
┌──────────────────────────────┐
│ ALFWorld Server (Python 3.6) │
│ - Manage environment         │
│ - Execute actions            │
│ - Return observations        │
└──────────────────────────────┘
```

## Files

- **`alfworld_server.py`** - ALFWorld environment server (Python 3.6+)
- **`alfworld_vllm_client.py`** - vLLM inference client (Python 3.11+)
- **`run_split_inference.py`** - Process manager to run both simultaneously

## Setup

### 1. Create Python Virtual Environments

```bash
# Create Python 3.6 environment for ALFWorld
python3.6 -m venv venv_py36
source venv_py36/bin/activate
pip install alfworld pyyaml pillow

# Create Python 3.11 environment for vLLM
python3.11 -m venv venv_py311
source venv_py311/bin/activate
pip install vllm transformers pyyaml pillow tqdm
```

### 2. Verify Installations

```bash
# Test Python 3.6 environment
source venv_py36/bin/activate
python -c "from alfworld.env.thor_env import ThorEnv; print('ALFWorld OK')"

# Test Python 3.11 environment
source venv_py311/bin/activate
python -c "from vllm import LLM; print('vLLM OK')"
```

## Usage

### Option 1: Using Process Manager (Recommended)

The easiest way to run both processes together:

```bash
# Activate any Python environment
source venv_py311/bin/activate  # or venv_py36

python run_split_inference.py \
    --server-python $(pwd)/../venv_py36/bin/python \
    --client-python $(pwd)/../venv_py311/bin/python \
    --model-path /root/checkpoints/global_step_795/ \
    --data-dir /root/data/alfworld/valid_seen/ \
    --traj-dir /root/data/alfworld/valid_seen_v5_high_trajs/ \
    --output-dir /root/data/alfworld/valid_seen_v5_vlm_inference_trajs/ \
    --num-samples 100
```

### Option 2: Manual Process Management

#### Terminal 1 - Start the ALFWorld Server

```bash
source venv_py36/bin/activate
python alfworld_server.py --host localhost --port 5555
```

Expected output:
```
[SERVER] ALFWorld server listening on localhost:5555
[SERVER] Client connected from ('127.0.0.1', 12345)
```

#### Terminal 2 - Start the vLLM Client

```bash
source venv_py311/bin/activate
python alfworld_vllm_client.py \
    --host localhost \
    --port 5555 \
    --model-path /root/checkpoints/global_step_795/ \
    --data-dir /root/data/alfworld/valid_seen/ \
    --traj-dir /root/data/alfworld/valid_seen_v5_high_trajs/ \
    --output-dir /root/data/alfworld/valid_seen_v5_vlm_inference_trajs/
```

## Communication Protocol

The client and server communicate via JSON messages over TCP sockets.

### Request Format

```json
{
    "command": "init|step|reset_task|get_state|is_goal_satisfied|shutdown",
    "...additional_args": "..."
}
```

### Response Format

```json
{
    "success": true,
    "...response_data": "..."
}
```

### Available Commands

#### `init` - Initialize a new task

**Request:**
```json
{"command": "init", "traj_file": "task_1.json"}
```

**Response:**
```json
{
    "success": true,
    "task_desc": "Pick up the apple",
    "intro": "You are in a room. Your task is to: Pick up the apple.",
    "initial_image": "base64_encoded_image"
}
```

#### `step` - Execute an action

**Request:**
```json
{"command": "step", "action": "go to table"}
```

**Response:**
```json
{
    "success": true,
    "action": "go to table",
    "feedback": "You go to the table.",
    "current_image": "base64_encoded_image",
    "action_success": true,
    "error_message": "",
    "goal_satisfied": false,
    "turn_count": 1
}
```

#### `is_goal_satisfied` - Check goal status

**Request:**
```json
{"command": "is_goal_satisfied"}
```

**Response:**
```json
{
    "success": true,
    "goal_satisfied": true
}
```

#### `get_state` - Get current state

**Request:**
```json
{"command": "get_state"}
```

**Response:**
```json
{
    "success": true,
    "turn_count": 5,
    "trajectory": ["initial obs", "action1", "feedback1", ...],
    "images_count": 6
}
```

#### `reset_task` - Reset for next task

**Request:**
```json
{"command": "reset_task"}
```

**Response:**
```json
{"success": true}
```

#### `shutdown` - Shutdown server

**Request:**
```json
{"command": "shutdown"}
```

**Response:**
```json
{"success": true, "message": "Server shutting down"}
```

## Customization

### Change Communication Port

```bash
python alfworld_server.py --port 6000
python alfworld_vllm_client.py --port 6000
```

### Change Host (for remote communication)

```bash
# Server on machine 1
python alfworld_server.py --host 0.0.0.0 --port 5555

# Client on machine 2
python alfworld_vllm_client.py --host machine1_ip --port 5555
```

### Extend Commands

To add new commands:

1. **Server side** (`alfworld_server.py`):
   ```python
   def cmd_my_command(self, request):
       """Implement your command logic"""
       return {'success': True, 'result': ...}

   # In process_request():
   elif command == 'my_command':
       return self.cmd_my_command(request)
   ```

2. **Client side** (`alfworld_vllm_client.py`):
   ```python
   def my_command(self):
       request = {'command': 'my_command', ...}
       return self.send_request(request)
   ```

## Troubleshooting

### Connection Refused
- Ensure server is running before starting client
- Check firewall settings
- Verify host and port match

### Socket Errors
- Increase timeout in `send_request()` if network is slow
- Check for multiple instances using the same port

### Large Image Transfer
If image transfer is slow, consider:
- Compressing images before encoding
- Using a faster compression format (e.g., JPEG)
- Modifying buffer sizes in socket communication

## Performance Considerations

1. **Network Overhead**: Each step requires JSON serialization/deserialization and socket I/O
   - Typical latency: 1-5ms per command
   - Negligible compared to model inference time

2. **Image Transfer**: Base64 encoding adds ~33% overhead
   - Consider JPEG compression for faster transfer
   - Images are sent one per step, not a bottleneck

3. **Socket Buffer Size**: Default 4096-8192 bytes
   - Increased to 8192 in client for image data
   - May need tuning for very large images

## Advantages of This Approach

✓ Separate Python versions (3.6 and 3.11)
✓ Easy to scale - run client and server on different machines
✓ Better debugging - can restart processes independently
✓ Flexible communication - easy to add monitoring/logging
✓ Simple protocol - JSON over sockets is easy to extend

## Converting Back to Single Process

If you resolve version conflicts, you can combine:
- Copy task initialization logic from `alfworld_server.py`
- Copy inference logic from `alfworld_vllm_client.py`
- Merge into a single script

## Additional Notes

- The original `alfworld_mm.py` has been preserved; this is a separate approach
- All image data is base64 encoded for JSON compatibility
- The server maintains state across multiple client requests
- Client handles model loading and inference independently

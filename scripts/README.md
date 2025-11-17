# Scripts Documentation

## Server Management Scripts (for ALFRED and ALFWorld-multimodal)

The `servers/` subdirectory contains scripts for managing X display servers used for running GUI applications in headless environments.

### Available Server Types

There are two types of display servers available:

1. **X Servers (Xorg) -- for GPUs** - Full X11 servers with hardware acceleration support
2. **Xvfb Servers -- for CPUs** - Virtual framebuffer X servers for software rendering

### Starting Servers

#### Start X Servers (Xorg) -- for GPUs

```bash
bash scripts/servers/start_x_servers.sh [num_displays]
```

**Details:**
- Starts displays from `:1` onwards (e.g., `:1`, `:2`, `:3`, `:4`)
- Automatically detects and skips already running displays
- Removes stale lock files before starting
- Logs output to `/tmp/xorg_display_*.log`

**Example:**
```bash
# Start 4 X servers
bash scripts/servers/start_x_servers.sh 4
```

#### Start Xvfb Servers --  for CPUs

```bash
bash scripts/servers/start_xvfb.sh [num_displays]
```

**Details:**
- Starts displays from `:11` onwards (e.g., `:11`, `:12`, `:13`, `:14`)
- Uses software rendering (LIBGL_ALWAYS_SOFTWARE=1)
- Screen resolution: 1024x768x24
- Automatically detects and skips already running displays
- Removes stale lock files before starting
- Logs output to `/tmp/xvfb_display_*.log`

**Example:**
```bash
# Start 4 Xvfb servers
bash scripts/servers/start_xvfb.sh 4
```

### Stopping Servers

#### Stop X Servers (Xorg)

```bash
bash scripts/servers/stop_x_servers.sh
```

#### Stop Xvfb Servers

```bash
bash scripts/servers/stop_xvfb.sh
```

### Monitoring and Troubleshooting

#### View Running Servers

```bash
# Check X servers
ps aux | grep Xorg

# Check Xvfb servers
ps aux | grep Xvfb
```

#### View Server Logs

```bash
# X server logs
tail -f /tmp/xorg_display_*.log

# Xvfb server logs
tail -f /tmp/xvfb_display_*.log

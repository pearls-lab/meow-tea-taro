#!/bin/bash
# Script to start multiple X servers in the background
# Usage: bash scripts/servers/start_x_servers.sh [num_displays]

NUM_DISPLAYS=${1:-4}  # Default to 4 displays if not specified
START_DISPLAY=1      # Start from display :1

for i in $(seq $START_DISPLAY $((START_DISPLAY + NUM_DISPLAYS - 1))); do
    # Check if display is already running
    if pgrep -f "Xorg.*:$i\$" > /dev/null; then
        echo "Display :$i is already running, skipping..."
        continue
    fi

    # Check for and remove stale lock files and sockets
    LOCK_FILE="/tmp/.X${i}-lock"
    if [ -f "$LOCK_FILE" ]; then
        echo "Found stale lock file $LOCK_FILE, removing..."
        rm -f "$LOCK_FILE"
    fi
    sleep 1

    # Start X server
    echo "Starting X server on display :$i..."
    nohup python meow_tea_gym/servers/start_x_server.py --display $i > /tmp/xorg_display_${i}.log 2>&1 &

    # Wait for X server to start (check log file or process)
    sleep 2

    # Get the actual Xorg PID
    XORG_PID=$(pgrep -f "Xorg.*:$i\$")
    if [ -n "$XORG_PID" ]; then
        echo "  ✓ Display :$i started (PID: $XORG_PID)"
    else
        echo "  ✗ Display :$i failed to start - check /tmp/xorg_display_${i}.log"
    fi
done

echo ""
echo "======================================================================"
echo "Summary of running X servers:"
ps aux | grep "[X]org" | awk '{printf "  Display :%s - PID: %s\n", substr($12,2), $2}'
echo "======================================================================"
echo ""
echo "Logs: /tmp/xorg_display_*.log"
echo "To view logs: tail -f /tmp/xorg_display_*.log"
echo "To stop all X servers: pkill Xorg"
echo ""

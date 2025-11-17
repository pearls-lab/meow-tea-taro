#!/bin/bash
# Script to start multiple Xvfb servers in the background
# Usage: bash scripts/servers/start_xvfb.sh [num_displays]

NUM_DISPLAYS=${1:-4}  # Default to 4 displays if not specified
START_DISPLAY=11      # Start from display :11

echo "Starting $NUM_DISPLAYS Xvfb servers..."
echo ""

# Set environment variables for software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

for i in $(seq $START_DISPLAY $((START_DISPLAY + NUM_DISPLAYS - 1))); do
    # Check if display is already running
    if pgrep -f "Xvfb.*:$i\b" > /dev/null; then
        echo "Display :$i is already running, skipping..."
        continue
    fi

    # Check for and remove stale lock files
    LOCK_FILE="/tmp/.X${i}-lock"
    if [ -f "$LOCK_FILE" ]; then
        echo "Found stale lock file $LOCK_FILE, removing..."
        rm -f "$LOCK_FILE"
    fi

    # Start Xvfb server
    echo "Starting Xvfb server on display :$i..."
    Xvfb :$i -screen 0 1024x768x24 -nolisten tcp -fbdir /tmp > /tmp/xvfb_display_${i}.log 2>&1 &

    # Wait for Xvfb server to start
    sleep 1

    # Get the actual Xvfb PID
    XVFB_PID=$(pgrep -f "Xvfb.*:$i\b")
    if [ -n "$XVFB_PID" ]; then
        echo "  ✓ Display :$i started (PID: $XVFB_PID)"
    else
        echo "  ✗ Display :$i failed to start - check /tmp/xvfb_display_${i}.log"
    fi
done

echo ""
echo "======================================================================"
echo "Summary of running Xvfb servers:"
ps aux | grep "[X]vfb" | awk '{printf "  Display :%s - PID: %s\n", substr($12,2), $2}'
echo "======================================================================"
echo ""
echo "Logs: /tmp/xvfb_display_*.log"
echo "To view logs: tail -f /tmp/xvfb_display_*.log"
echo "To stop all Xvfb servers: pkill Xvfb"
echo ""

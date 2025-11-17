#!/bin/bash
# Script to stop all Xvfb servers
# Usage: bash scripts/servers/stop_xvfb.sh

echo "Stopping all Xvfb servers..."
echo ""

# Get list of running Xvfb processes
XVFB_PIDS=$(pgrep -f "Xvfb")

if [ -z "$XVFB_PIDS" ]; then
    echo "No Xvfb servers are currently running."
    echo "Cleaning up any stale lock files..."
else
    # Show what will be stopped
    echo "Found the following Xvfb servers:"
    ps aux | grep "Xvfb" | awk '{printf "  Display :%s - PID: %s\n", substr($12,2), $2}'
    echo ""

    # Stop them
    pkill Xvfb

    sleep 1

    # Verify they're stopped
    REMAINING=$(pgrep -f "Xvfb")
    if [ -z "$REMAINING" ]; then
        echo "✓ All Xvfb servers stopped successfully"
    else
        echo "⚠ Some Xvfb servers are still running. Forcing..."
        pkill -9 Xvfb
        sleep 1
        STILL_REMAINING=$(pgrep -f "Xvfb")
        if [ -z "$STILL_REMAINING" ]; then
            echo "✓ All Xvfb servers forcefully stopped"
        else
            echo "✗ Failed to stop some Xvfb servers:"
            ps aux | grep "[X]vfb"
        fi
    fi
fi

# Clean up lock files
echo ""
echo "Cleaning up lock files..."
CLEANED=0

for i in {1..30}; do
    LOCK_FILE="/tmp/.X${i}-lock"

    if [ -f "$LOCK_FILE" ]; then
        rm -f "$LOCK_FILE"
        echo "  Removed lock file: $LOCK_FILE"
        CLEANED=$((CLEANED + 1))
    fi
done

if [ $CLEANED -eq 0 ]; then
    echo "  No stale files found"
else
    echo "  Cleaned $CLEANED stale file(s)"
fi

echo ""
echo "✓ Cleanup complete"
echo ""

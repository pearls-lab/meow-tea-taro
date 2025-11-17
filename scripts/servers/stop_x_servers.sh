#!/bin/bash
# Script to stop all X servers
# Usage: bash scripts/servers/stop_x_servers.sh

echo "Stopping all X servers..."
echo ""

# Get list of running X servers
XORG_PIDS=$(pgrep -f "Xorg")

if [ -z "$XORG_PIDS" ]; then
    echo "No X servers are currently running."
    echo "Cleaning up any stale lock files..."
else
    # Show what will be stopped
    echo "Found the following X servers:"
    ps aux | grep "Xorg" | awk '{printf "  Display :%s - PID: %s\n", substr($12,2), $2}'
    echo ""

    # Stop them
    pkill Xorg

    sleep 1

    # Verify they're stopped
    REMAINING=$(pgrep -f "Xorg")
    if [ -z "$REMAINING" ]; then
        echo "✓ All X servers stopped successfully"
    else
        echo "⚠ Some X servers are still running. Forcing..."
        pkill -9 Xorg
        sleep 1
        STILL_REMAINING=$(pgrep -f "Xorg")
        if [ -z "$STILL_REMAINING" ]; then
            echo "✓ All X servers forcefully stopped"
        else
            echo "✗ Failed to stop some X servers:"
            ps aux | grep "[X]org"
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

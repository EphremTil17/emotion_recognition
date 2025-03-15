#!/bin/bash
# kill_ports.sh
# This script forcefully kills any process listening on ports 8000, 8001, and 8002,
# and also kills the Python services using the PID files.

# Define the script directory and PID directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$SCRIPT_DIR/pids"

# Define an array of ports to kill processes on
ports=(8000 8001 8002 8003)  # Added port 8003 in case scholarly_search uses it

# Kill processes by port
for port in "${ports[@]}"; do
  echo "Checking port $port..."
  # Use lsof to list process IDs (-t: terse output) of processes listening on TCP port.
  pids=$(lsof -t -i tcp:"$port")
  if [ -n "$pids" ]; then
    echo "Processes found on port $port: $pids"
    # Force-kill all processes found.
    kill -9 $pids
    echo "Killed processes on port $port."
  else
    echo "No processes found on port $port."
  fi
done

# Kill processes by PID file
services=("processing_service" "realtime_service" "content_analysis" "scholarly_search" "npm_run_dev")

for service in "${services[@]}"; do
  pid_file="${PID_DIR}/${service}.pid"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    echo "Stopping $service (PID: $pid)..."
    # Check if process exists before killing it
    if ps -p "$pid" > /dev/null; then
      kill -9 "$pid"
      echo "Killed $service process"
    else
      echo "$service process was not running"
    fi
    # Remove the PID file
    rm "$pid_file"
  else
    echo "No PID file found for $service"
  fi
done

echo "All services have been stopped"
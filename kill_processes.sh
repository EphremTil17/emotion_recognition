#!/bin/bash
# kill_ports.sh
# This script forcefully kills any process listening on ports 8000, 8001, and 8002.

# Define an array of ports.
ports=(8000 8001 8002)

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

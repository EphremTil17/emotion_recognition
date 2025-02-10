#!/bin/bash
# run_all.sh

# Get the directory where the script is located.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/log"
PID_DIR="$SCRIPT_DIR/pids"

# Create the log and PID directories if they don't exist.
mkdir -p "$LOG_DIR" "$PID_DIR"

# Function to start a service in a specified directory.
start_service() {
  local service_name=$1    # Unique name for the service.
  local cmd=$2             # Command to run.
  local working_dir=$3     # Relative directory in which to run the command.
  local pid_file="$PID_DIR/${service_name}.pid"

  # Change to the specified working directory.
  if [ -n "$working_dir" ]; then
    cd "$SCRIPT_DIR/$working_dir" || { echo "Cannot cd to $working_dir"; exit 1; }
  fi

  # Start the command in background using nohup and redirect output to a log file.
  nohup bash -c "$cmd" > "$LOG_DIR/${service_name}.log" 2>&1 &
  local pid=$!
  echo $pid > "$pid_file"
  echo "Started $service_name with PID $pid"

  # Return to the base script directory.
  cd "$SCRIPT_DIR" || exit 1
}

# Start Python processing_service from the backend directory.
start_service "processing_service" "python3 processing_service.py" "backend"

# Start Python realtime_service from the backend directory.
start_service "realtime_service" "python3 realtime_service.py" "backend"

# Start the npm dev server from the frontend directory.
start_service "npm_run_dev" "npm run dev" "frontend"

echo "All services started. Logs are in the 'log' directory, and PIDs are stored in 'pids'."

# Optionally, wait for all background processes (this will block the terminal).
# Uncomment the next line if you want the script to remain active:
# wait

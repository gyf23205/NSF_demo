import subprocess
import time

# Define the scripts and their arguments
scripts = {
    "script1": ["python", "ecg_record.py"],
    "script2": ["python", "online_data_collector.py", "--gaze-dir",
                "C:/Users/JW Choi/Desktop/WE_transformer/data_collection/openface", "--ecg-csv",
                "C:/Users/JW Choi/Desktop/NSF_2025_demo/dataset/JW_record/ecg_log.csv"],
    "script3": ["python", "data_aggregator.py", "--gaze-csv", "C:/Users/JW Choi/Desktop/NSF_2025_demo/dataset/gaze_cleaned.csv", 
                "--ecg-csv", "C:/Users/JW Choi/Desktop/NSF_2025_demo/dataset/JW_record/ecg_log.csv"]
            }

# Store subprocess handles
processes = {}

# Function to start a script
def start_script(name, command):
    print(f"Starting {name} with command: {' '.join(command)}")
    return subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)

# Start all scripts
for name, command in scripts.items():
    processes[name] = start_script(name, command)

try:
    while True:
        for name, process in processes.items():
            if process.poll() is not None:  # Process has exited
                print(f"--------------------------------------------{name} has stopped. Restarting...")
                processes[name] = start_script(name, scripts[name])
        time.sleep(2)
except KeyboardInterrupt:
    print("----------------------------------Stopping all scripts...--------------------------------------------")
    for process in processes.values():
        process.terminate()

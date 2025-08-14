import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

folder = "C:\\Users\\JW Choi\\Desktop\\NSF_demo-main\\NSF_demo\\examples\\workload_history"

# Find the most recent CSV in workload_history/ with wl_history_*.csv pattern
files = [f for f in os.listdir(folder) if f.startswith("wl_history_") and f.endswith(".csv")]
if not files:
    raise FileNotFoundError(f"No CSV files found in {folder} matching wl_history_*.csv")

latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(folder, f)))
latest_path = os.path.join(folder, latest_file)
print(f"Reading from: {latest_path}")

# Read timestamps and values
timestamps = []
values = []
with open(latest_path, newline="") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # skip header
    for row in reader:
        if len(row) < 2:
            continue
        try:
            timestamps.append(datetime.fromisoformat(row[0]))
            if float(row[1]) > 1:
                values.append(1.0)
            elif float(row[1]) < 0:
                values.append(0.0)
            else:
                values.append(float(row[1]))
        except ValueError:
            continue

if not values:
    raise ValueError("No valid data found in file.")

# Convert timestamps to seconds relative to the first timestamp
start_time = timestamps[0]
times_sec = [(t - start_time).total_seconds() for t in timestamps]

# Plot values over time in seconds
plt.plot(times_sec, values, label="Value")
plt.xlabel("Time [s]")
plt.ylabel("Estimated workload")
plt.title("Workload Over Time (1: High, 0: Low)")
plt.legend()
plt.tight_layout()
plt.show()

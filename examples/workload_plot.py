import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

folder = "workload_history"

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
            values.append(float(row[1]))
        except ValueError:
            continue

if not values:
    raise ValueError("No valid data found in file.")

# Compute running average
running_avg = []
total = 0.0
for i, val in enumerate(values, start=1):
    total += val
    running_avg.append(total / i)

overall_avg = total / len(values)
print(f"Overall average: {overall_avg:.6f}")

# Plot running average over time
plt.plot(timestamps, running_avg, label="Running average")
plt.xlabel("Time")
plt.ylabel("Average value")
plt.title("Running Average Over Time")
plt.legend()
plt.tight_layout()
plt.show()

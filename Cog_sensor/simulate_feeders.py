import time
import pandas as pd
import multiprocessing

def simulate_gaze_feeder(source_csv, target_csv, interval=1.0):
    df = pd.read_csv(source_csv)
    print(f"[Gaze Feeder] Starting simulation with {len(df)} rows...")
    
    with open(target_csv, 'w') as f:
        df.iloc[[0]].to_csv(f, index=False)  # Write header and first row

    for i in range(1, len(df)):
        time.sleep(interval)
        with open(target_csv, 'a') as f:
            df.iloc[[i]].to_csv(f, index=False, header=False)
        print(f"[Gaze Feeder] Added row {i}/{len(df)}")

def simulate_ecg_feeder(source_csv, target_csv, interval=1.0):
    df = pd.read_csv(source_csv)
    print(f"[ECG Feeder] Starting simulation with {len(df)} rows...")

    buffer = []

    for i in range(len(df)):
        time.sleep(interval)

        # Convert row to dictionary
        row_dict = df.iloc[i].to_dict()
        buffer.append(row_dict)

        # Keep only the most recent 2 rows
        if len(buffer) > 2:
            buffer = buffer[-2:]

        # Overwrite the file
        pd.DataFrame(buffer).to_csv(target_csv, index=False, header=False)

        print(f"[ECG Feeder] Updated with row {i}/{len(df)} (keeping {len(buffer)} rows)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate Gaze and ECG data streaming.")
    parser.add_argument("--gaze-source", type=str, required=True, help="Path to full Gaze CSV file")
    parser.add_argument("--gaze-target", type=str, default="gaze_log.csv", help="Path to simulated Gaze output CSV")
    parser.add_argument("--ecg-source", type=str, required=True, help="Path to full ECG CSV file")
    parser.add_argument("--ecg-target", type=str, default="ecg_log.csv", help="Path to simulated ECG output CSV")
    parser.add_argument("--gaze-interval", type=float, default=1.0, help="Gaze sampling interval (seconds)")
    parser.add_argument("--ecg-interval", type=float, default=1.0, help="ECG sampling interval (seconds)")
    args = parser.parse_args()

    # Create two processes
    gaze_proc = multiprocessing.Process(target=simulate_gaze_feeder, args=(args.gaze_source, args.gaze_target, args.gaze_interval))
    ecg_proc = multiprocessing.Process(target=simulate_ecg_feeder, args=(args.ecg_source, args.ecg_target, args.ecg_interval))

    gaze_proc.start()
    ecg_proc.start()

    gaze_proc.join()
    ecg_proc.join()
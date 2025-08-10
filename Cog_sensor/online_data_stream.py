# This script run real-time data collection and aggregation and produces a single fused vector per window (ECG + gaze)

import pandas as pd
import numpy as np
import os
import time
import csv
from collections import deque

class OnlineGazeExtractor:
    def __init__(self, csv_path, output_path, max_rows=50, poll_interval=1.0):
        self.csv_path = csv_path
        self.output_path = output_path
        self.last_index = 0
        self.max_rows = max_rows         # Number of rows to retain
        self.poll_interval = poll_interval  # Seconds between polls


    def extract_features(self, df_new):
        # Clean both index and data
        df_new.columns = df_new.columns.str.strip()
        df_new.rename(columns=lambda x: x.strip(), inplace=True)  # extra guard

        # print("[debug] Cleaned columns:", list(df_new.columns))

        # required_cols = [
        #     'timestamp',
        #     'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
        #     'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
        #     'gaze_angle_x', 'gaze_angle_y'
        # ]

        required_cols = [
            'timestamp',
            'gaze_angle_x', 'gaze_angle_y', 
            'AU04_r', 'AU05_r', 'AU09_r', 'AU10_r', 'AU14_r', 
            'AU15_r', 'AU17_r', 'AU04_c'
        ]

        # Verify all columns exist before selecting
        missing = [col for col in required_cols if col not in df_new.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        return df_new[required_cols]

    def run(self):
        print("Starting online gaze extraction...")
        while True:
            try:
                df = pd.read_csv(self.csv_path)
                df.columns = df.columns.str.strip()
                # print(df.columns)
                new_rows = df.iloc[self.last_index:].copy()

                extracted = self.extract_features(new_rows)
                if extracted is not None and not extracted.empty:
                    # Append extracted to output
                    extracted.to_csv(self.output_path, mode='a', header=not self.last_index, index=False)
                    self.last_index += len(new_rows)

                    print(f"[GazeExtractor] Extracted and saved {len(extracted)} new rows")

                    # --------- Truncate the cleaned output (gaze_cleaned.csv) ----------
                    try:
                        df_out = pd.read_csv(self.output_path)
                        if len(df_out) > self.max_rows:
                            #print(f"[debug] Truncating output CSV from {len(df_out)} rows to {self.max_rows}")
                            df_out = df_out.iloc[-self.max_rows:]
                            df_out.to_csv(self.output_path, index=False)
                    except Exception as e:
                        print(f"[GazeExtractor] Warning during output truncation: {e}")
                    # ---------------------------------------------------------------------

                # --------- Truncate the input (gaze_log.csv) ----------
                if len(df) > self.max_rows:
                    #print(f"[debug] Truncating input CSV from {len(df)} rows to {self.max_rows}")
                    df = df.iloc[-self.max_rows:]
                    df.to_csv(self.csv_path, index=False)
                    self.last_index = min(self.last_index, self.max_rows)
                # ------------------------------------------------------

                time.sleep(self.poll_interval)

            except Exception as e:
                print(f"Error during polling: {e}")
                time.sleep(2)

class OnlineECGBuffer:
    def __init__(self, csv_path, buffer_duration=3, sampling_rate=130):
        self.csv_path = csv_path
        self.buffer_duration = buffer_duration
        self.sampling_rate = sampling_rate
        self.samples_per_row = 73
        self.max_rows = int(np.ceil((buffer_duration * sampling_rate) / self.samples_per_row))
        self.buffer = deque(maxlen=self.max_rows)



    def update(self):
        """Reads latest rows from ECG CSV and updates the buffer."""
        print("[ECGBuffer] update() called")
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, 'r') as f:
                reader = list(csv.reader(f))

            # Parse only the last N rows
            for row in reader[-self.max_rows:]:
                if len(row) < 2 or row[1].strip() == "":
                    continue  # skip empty or malformed rows

                try:
                    signal_str = row[1]
                    signal = np.array(eval(signal_str), dtype=np.float32)
                    if signal.shape[0] > 0:
                        self.buffer.append(signal)
                except Exception as e:
                    print(f"[ECGBuffer] Failed to parse row: {e}")
                    continue

        except Exception as e:
            print(f"[ECGBuffer] Error reading ECG file: {e}")

        print(f"[ECGBuffer] Buffer Size: {len(self.buffer)}")

    def get_buffer(self):
        """Returns the most recent ECG buffer as a 1D array."""
        if len(self.buffer) == 0:
            print("[WARNING] ECG buffer is empty!")
            return None
        stacked = np.concatenate(self.buffer)
        print(f"[DEBUG] Final ECG vector shape: {stacked.shape}")
        return stacked[-int(self.buffer_duration * self.sampling_rate):]  # Ensure exact duration
    
    

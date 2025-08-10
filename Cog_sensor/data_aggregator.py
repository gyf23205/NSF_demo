import numpy as np
import pandas as pd
import time
from scipy import interpolate

class DataAggregator:
    def __init__(self, ecg_buffer, gaze_csv_path, window_duration=3.0, gaze_poll_rate=0.5, output_csv="aggregated_output.csv", ecg_target_length=130):
        self.ecg_buffer = ecg_buffer
        self.gaze_csv_path = gaze_csv_path
        self.window_duration = window_duration
        self.last_gaze_timestamp = 0.0
        self.prev_gaze_vector = np.zeros(8) 
        self.gaze_poll_rate = gaze_poll_rate
        self.output_csv = output_csv
        self.first_write = True
        self.ecg_target_length = ecg_target_length  

    def get_latest_gaze(self, N=1.5, L=30):
        try:
            df = pd.read_csv(self.gaze_csv_path)

            # Keep only rows newer than the last timestamp
            df = df[df['timestamp'] >= self.last_gaze_timestamp]

            if df.empty:
                return self.prev_gaze_vector

            # Get the last timestamp in the filtered data
            latest_time = df['timestamp'].max()
            df_window = df[df['timestamp'] >= latest_time - N]

            # Sanity check: is the actual time window close enough to the expected N?
            actual_window_start = df_window['timestamp'].min()
            actual_window_duration = latest_time - actual_window_start
            allowed_deviation = 0.3  # seconds (or whatever threshold you like)

            if abs(actual_window_duration - N) > allowed_deviation:
                print(f"[Warning] Gaze data window duration {actual_window_duration:.2f}s deviates from target {N:.2f}s by more than {allowed_deviation:.2f}s.")
            else:
                print('good data')


            if df_window.empty or len(df_window) < 2:
                # Not enough data to interpolate
                return self.prev_gaze_vector

            # Define the features to include
            features = [
                'gaze_angle_x', 'gaze_angle_y',
                'AU04_r', 'AU05_r', 'AU09_r', 'AU10_r',
                'AU14_r', 'AU15_r', 'AU17_r', 'AU04_c'
            ]

            timestamps = df_window['timestamp'].values
            interpolated = []

            # Interpolate each feature independently
            for feat in features:
                values = df_window[feat].values
                interp_fn = interpolate.interp1d(
                    timestamps,
                    values,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )

                # Uniform time steps for interpolation
                target_times = np.linspace(timestamps[0], timestamps[-1], L)
                interpolated_feat = interp_fn(target_times)
                interpolated.append(interpolated_feat)

            # Stack into shape (L, num_features)
            gaze_tensor = np.stack(interpolated, axis=1).astype(np.float32)

            self.prev_gaze_vector = gaze_tensor
            self.last_gaze_timestamp = latest_time + self.gaze_poll_rate

            return gaze_tensor.T.flatten()

        except Exception as e:
            print(f"Gaze read error: {e}")
            return self.prev_gaze_vector


    # def get_latest_gaze(self): # Old version, bf 10 Jul 2025
    #     try:
    #         df = pd.read_csv(self.gaze_csv_path)
    #         df = df[df['timestamp'] >= self.last_gaze_timestamp]
    #         if df.empty:
    #             return self.prev_gaze_vector

    #         row = df.iloc[-1]
    #         # gaze_vector = np.array([
    #         #     row['gaze_0_x'], row['gaze_0_y'], row['gaze_0_z'],
    #         #     row['gaze_1_x'], row['gaze_1_y'], row['gaze_1_z'],
    #         #     row['gaze_angle_x'], row['gaze_angle_y']
    #         # ], dtype=np.float32)

    #         gaze_vector = np.array([
    #             row['gaze_angle_x'], row['gaze_angle_y'], row['AU04_r'],
    #             row['AU05_r'], row['AU09_r'], row['AU10_r'],
    #             row['AU14_r'], row['AU15_r'], row['AU17_r'], row['AU04_c']
    #         ], dtype=np.float32)

    #         self.prev_gaze_vector = gaze_vector
    #         self.last_gaze_timestamp = row['timestamp'] + self.gaze_poll_rate
    #         return gaze_vector
    #     except Exception as e:
    #         print(f"Gaze read error: {e}")
    #         return self.prev_gaze_vector

    def resample_ecg(self, ecg_vector):
        """Resample the ECG vector to fixed target length."""
        original_length = len(ecg_vector)

        if original_length == 0:
            return np.zeros(self.ecg_target_length, dtype=np.float32)

        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, self.ecg_target_length)
        f = interpolate.interp1d(x_old, ecg_vector, kind='linear', fill_value="extrapolate")
        ecg_resampled = f(x_new).astype(np.float32)

        return ecg_resampled

    def get_aggregated_vector(self):
        self.ecg_buffer.update()
        ecg_vector = self.ecg_buffer.get_buffer()
        gaze_vector = self.get_latest_gaze()

        if ecg_vector is None or len(ecg_vector) == 0:
            ecg_vector = np.zeros(int(self.ecg_buffer.buffer_duration * self.ecg_buffer.sampling_rate), dtype=np.float32)

        ecg_vector = self.resample_ecg(ecg_vector)

        print(np.shape(ecg_vector))
        print(np.shape(gaze_vector))
        print(np.shape(np.concatenate([ecg_vector, gaze_vector], axis=0)))

        return np.concatenate([ecg_vector, gaze_vector], axis=0)

    def run_loop(self):
        print("Starting data aggregator loop...")
        while True:
            fused_vec = self.get_aggregated_vector()

            # Save fused vector
            df = pd.DataFrame(fused_vec.reshape(1, -1))  # (1, n)
            df.to_csv(self.output_csv, mode='a', index=False, header=self.first_write)

            if self.first_write:
                self.first_write = False

            print(f"[Aggregator] Saved vector of shape {fused_vec.shape} to {self.output_csv}")

            time.sleep(self.window_duration)

if __name__ == "__main__":
    import argparse
    from online_data_stream import OnlineECGBuffer

    parser = argparse.ArgumentParser(description="Run data aggregator.")
    parser.add_argument("--gaze-csv", type=str, required=True)
    parser.add_argument("--ecg-csv", type=str, required=True)
    parser.add_argument("--window-duration", type=float, default=3.0)
    parser.add_argument("--gaze-poll-rate", type=float, default=0.5)
    parser.add_argument("--output-csv", type=str, default="aggregated_output.csv")
    parser.add_argument("--ecg-target-length", type=int, default=130)

    args = parser.parse_args()

    # Set up ECG buffer
    ecg_buffer = OnlineECGBuffer(
        csv_path=args.ecg_csv,
        buffer_duration=args.window_duration,
        sampling_rate=130
    )

    # Set up and run aggregator
    aggregator = DataAggregator(
        ecg_buffer=ecg_buffer,
        gaze_csv_path=args.gaze_csv,
        window_duration=args.window_duration,
        gaze_poll_rate=args.gaze_poll_rate,
        output_csv=args.output_csv,
        ecg_target_length=args.ecg_target_length
    )
    aggregator.run_loop()
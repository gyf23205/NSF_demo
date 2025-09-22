import numpy as np
import pandas as pd
import time
from scipy import interpolate
from scipy.signal import butter, filtfilt, find_peaks


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

    # def get_aggregated_vector(self):
    #     self.ecg_buffer.update()
    #     ecg_vector = self.ecg_buffer.get_buffer()
    #     gaze_vector = self.get_latest_gaze()

    #     if ecg_vector is None or len(ecg_vector) == 0:
    #         ecg_vector = np.zeros(int(self.ecg_buffer.buffer_duration * self.ecg_buffer.sampling_rate), dtype=np.float32)

    #     ecg_vector = self.resample_ecg(ecg_vector)

    #     print(np.shape(ecg_vector))
    #     print(np.shape(gaze_vector))
    #     print(np.shape(np.concatenate([ecg_vector, gaze_vector], axis=0)))

    #     return np.concatenate([ecg_vector, gaze_vector], axis=0)

    def _bandpass(x, fs, low, high, order):
        nyq = 0.5 * fs
        # clamp high below Nyquist, ensure low < high; else skip filtering
        high = min(high, 0.9 * nyq)
        if high <= 0:
            return x
        low = max(0.1, min(low, high * 0.8))
        if low >= high:
            return x
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, x)
    
    def extract_hrv_features(signal, fs, bandpass=None, order=3,
                            min_distance_s=0.30, prom_factor=0.8,
                            return_nan_on_fail=True):
        """
        Time-domain HRV features from a single ECG segment.

        Parameters
        ----------
        signal : np.ndarray
            ECG segment (1D).
        fs : float
            Sampling frequency (Hz) of *this segment*.
        bandpass : (low, high) in Hz or None
            If not None, band-pass before peak detection. If invalid vs fs, it is auto-clamped.
        order : int
            IIR filter order when bandpass is used.
        min_distance_s : float
            Minimum RR (s) to avoid double detections (~0.30s -> ~200 bpm max).
        prom_factor : float
            Peak prominence = prom_factor * std(filtered_signal).
        return_nan_on_fail : bool
            If True, returns NaNs on failure; else returns zeros.

        Returns
        -------
        np.ndarray shape (4,) dtype float32
            [HR (bpm), SDNN (s), RMSSD (s), pNN50 (%)]
            Note: SDNN/RMSSD are in seconds here (consistent with RR in seconds).
        """

        # --- guards ---
        if signal is None or len(signal) < 4 or fs <= 0:
            v = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
            return v if return_nan_on_fail else np.zeros(4, dtype=np.float32)

        x = np.asarray(signal, dtype=float)

        # --- optional filtering for more reliable R-peaks ---
        if bandpass is not None and isinstance(bandpass, (tuple, list)) and len(bandpass) == 2:
            low, high = float(bandpass[0]), float(bandpass[1])
            x_filt = bandpass(x, fs, low, high, order)
        else:
            x_filt = x

        # --- R-peak detection (try both polarity if needed) ---
        distance = max(1, int(min_distance_s * fs))
        std = np.std(x_filt)
        prominence = (prom_factor * std) if std > 0 else None

        peaks, _ = find_peaks(x_filt, distance=distance, prominence=prominence)
        if len(peaks) < 2:
            # try inverted signal (in case QRS is negative)
            peaks, _ = find_peaks(-x_filt, distance=distance, prominence=prominence)
            if len(peaks) < 2:
                v = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
                return v if return_nan_on_fail else np.zeros(4, dtype=np.float32)

        # --- RR intervals (seconds) ---
        rr = np.diff(peaks) / float(fs)  # seconds
        if rr.size == 0:
            v = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)
            return v if return_nan_on_fail else np.zeros(4, dtype=np.float32)

        # --- time-domain features ---
        rr_mean = rr.mean()
        HR = 60.0 / rr_mean if rr_mean > 0 else np.nan

        SDNN = np.std(rr, ddof=1) if rr.size > 1 else np.nan

        drr = np.diff(rr)
        RMSSD = np.sqrt(np.mean(drr ** 2)) if drr.size > 0 else np.nan

        NN50 = np.sum(np.abs(drr) > 0.05) if drr.size > 0 else 0  # 50 ms = 0.05 s
        pNN50 = (100.0 * NN50 / drr.size) if drr.size > 0 else np.nan

        out = np.array([HR, SDNN, RMSSD, pNN50], dtype=np.float32)

        # if user wants zeros instead of NaNs on short/poor segments
        if not return_nan_on_fail and (np.isnan(out).any()):
            out = np.nan_to_num(out, nan=0.0).astype(np.float32)

        return out


    def get_aggregated_vector(self):
        self.ecg_buffer.update()
        ecg_vector = self.ecg_buffer.get_buffer()
        gaze_vector = self.get_latest_gaze()

        if ecg_vector is None or len(ecg_vector) == 0:
            # fallback ECG if empty
            ecg_resampled = np.zeros(self.ecg_target_length, dtype=np.float32)
        else:
            # resample ECG to fixed length
            ecg_resampled = self.resample_ecg(ecg_vector)

        # compute features from resampled ECG
        fs_effective = self.ecg_target_length / self.window_duration
        ecg_features = extract_hrv_features(ecg_resampled,fs_effective)
        if ecg_features is None:
            ecg_features = np.zeros(4, dtype=np.float32)  # HR, SDNN, RMSSD, pNN50

        # 🚨 stack = raw ECG + features + gaze
        fused = np.concatenate([ecg_resampled, ecg_features, gaze_vector], axis=0)

        print("ECG raw shape:", ecg_resampled.shape)
        print("ECG features shape:", ecg_features.shape)
        print("Gaze vector shape:", gaze_vector.shape)
        print("Final fused vector:", fused.shape)

        return fused

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
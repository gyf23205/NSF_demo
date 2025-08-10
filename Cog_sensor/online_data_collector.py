import threading
import time
import os
from online_data_stream import OnlineGazeExtractor, OnlineECGBuffer

def get_latest_gaze_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSV files found in directory.")
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return os.path.join(directory, latest)

def start_gaze_extraction(csv_path, output_path, max_rows=50, poll_interval=1.0):
    gaze_extractor = OnlineGazeExtractor(
        csv_path=csv_path,
        output_path=output_path,
        max_rows=max_rows,
        poll_interval=poll_interval
    )
    gaze_extractor.run()

def start_ecg_buffer(csv_path, buffer_duration=3, sampling_rate=130):
    ecg_buffer = OnlineECGBuffer(
        csv_path=csv_path,
        buffer_duration=buffer_duration,
        sampling_rate=sampling_rate
    )
    while True:
        ecg_buffer.update()
        ecg_array = ecg_buffer.get_buffer()
        if ecg_array is not None:
            print(f"[ECGBuffer] Buffer shape: {ecg_array.shape}")
        time.sleep(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run online gaze and ECG collection together.")
    parser.add_argument("--gaze-csv", type=str, required=False, help="Path to the simulated gaze input file (e.g., gaze_log.csv)")# use this with simulated data
    parser.add_argument("--gaze-dir", type=str, required=True, help="Directory where OpenFace outputs .csv files")# use this with real OpenFace data
    parser.add_argument("--gaze-output", type=str, default="gaze_cleaned.csv", help="Path to cleaned gaze output file")
    parser.add_argument("--ecg-csv", type=str, required=True, help="Path to the simulated ECG input file (e.g., ecg_log.csv)")
    args = parser.parse_args()
    
    # python online_data_collector.py --gaze-dir "C:\Users\JW Choi\Desktop\WE_transformer\data_collection\openface" --ecg-csv "C:\Users\JW Choi\Desktop\NSF_2025_demo\dataset\JW_record\ecg_log.csv"

    if args.gaze_csv:
        gaze_csv_path = args.gaze_csv
    else:
        gaze_csv_path = get_latest_gaze_file(args.gaze_dir)

    # Launch Gaze extractor thread
    gaze_thread = threading.Thread(target=start_gaze_extraction, args=(gaze_csv_path, args.gaze_output))
    gaze_thread.start()

    # Run ECG buffer updater (main thread)
    start_ecg_buffer(args.ecg_csv)
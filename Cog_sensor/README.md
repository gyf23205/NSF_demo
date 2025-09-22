The documetation can be found: 

https://docs.google.com/document/d/1j5IbG8wSAcjx_HJaLmaTNU46Rn7IKsBRcXv_aWcbsFw/edit?tab=t.0

Branch created by JW Choi at 09 June 2025

Terminal 1:
python ecg_record.py

Terminal 2: 
python online_data_collector.py --gaze-dir "[YOUR OPENFACE CSV LOCATION]" --ecg-csv "[YOUR ECG CSV LOCATION]"


Terminal 3:
python data_aggregator_feature.py --gaze-csv gaze_cleaned.csv --ecg-csv "[YOUR ECG CSV LOCATION]"
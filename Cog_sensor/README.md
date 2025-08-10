The documetation can be found: 

https://docs.google.com/document/d/1j5IbG8wSAcjx_HJaLmaTNU46Rn7IKsBRcXv_aWcbsFw/edit?tab=t.0

Branch created by JW Choi at 09 June 2025

Terminal 1:
python ecg_record.py

Terminal 2: 
python online_data_collector.py --gaze-dir "C:\Users\JW Choi\Desktop\WE_transformer\data_collection\openface" --ecg-csv "C:\Users\JW Choi\Desktop\NSF_2025_demo\dataset\JW_record\ecg_log.csv"

python online_data_collector.py --gaze-dir "C:\Users\JW Choi\Desktop\NSF_2025_demo\dataset\JW_record" --ecg-csv "C:\Users\JW Choi\Desktop\NSF_2025_demo\dataset\JW_record\ecg_log.csv"

Terminal 3:
python data_aggregator.py --gaze-csv gaze_cleaned.csv --ecg-csv "C:\Users\JW Choi\Desktop\NSF_2025_demo\dataset\JW_record\ecg_log.csv"
import pandas as pd

# CSV'leri yükle
unity_log = pd.read_csv('DistanceLog.csv', sep=';')
python_log = pd.read_csv('apriltag_log.csv')

# Zaman sütunlarını datetime objesine çevir
unity_log['SystemTime'] = pd.to_datetime(unity_log['SystemTime'])
python_log['Time'] = pd.to_datetime(python_log['Time'])

# Farkı hesapla
time_diffs = []
for i in range(min(len(unity_log), len(python_log))):
    delta = abs((unity_log['SystemTime'][i] - python_log['Time'][i]).total_seconds())
    time_diffs.append(delta)

print("Ortalama zaman farkı (saniye):", sum(time_diffs)/len(time_diffs))

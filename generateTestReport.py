import pandas as pd
import numpy as np
import csv

def analyze_log(log_file, version, camera_index, filter_status):
    df = pd.read_csv(log_file)

    # FPS hesapla
    timestamps = df['Time'].values
    fps_values = 1 / np.diff(timestamps)
    avg_fps = np.mean(fps_values)

    # Z ekseni pozisyon kararlılığı
    tz_std = np.std(df['Tz'])

    # Decision Margin istatistikleri
    dm = df['DecisionMargin']
    dm_mean = np.mean(dm)
    dm_std = np.std(dm)

    return {
        "Versiyon": version,
        "Kamera": camera_index,
        "Filtre": filter_status,
        "FPS Ort.": round(avg_fps, 2),
        "Tz Std Sapma": round(tz_std, 4),
        "DM Ort.": round(dm_mean, 2),
        "DM Std": round(dm_std, 2)
    }

# === Test Senaryoları ===
test_cases = [
    {"log": "headpose_log.csv", "version": "HeadPose", "camera": 0, "filter": "ON"},
    {"log": "apriltag_log.csv", "version": "AprilTag", "camera": 0, "filter": "ON"},
    {"log": "headpose_log.csv", "version": "HeadPose", "camera": 0, "filter": "OFF"},
    {"log": "apriltag_log.csv", "version": "AprilTag", "camera": 0, "filter": "OFF"},
]

# === Raporu Oluştur
report_rows = []
for case in test_cases:
    result = analyze_log(case["log"], case["version"], case["camera"], case["filter"])
    report_rows.append(result)


# === CSV'ye Yaz
report_file = "test_report.csv"
with open(report_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=report_rows[0].keys())
    writer.writeheader()
    writer.writerows(report_rows)

print(f"\n✅ Test raporu oluşturuldu: {report_file}")


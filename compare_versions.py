
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Dosya Yolları ===
HEADPOSE_LOG = "headpose_log.csv"
APRILTAG_LOG = "apriltag_log.csv"

# === CSV'leri Oku ===
df_head = pd.read_csv(HEADPOSE_LOG)
df_april = pd.read_csv(APRILTAG_LOG)

# === FPS Hesapla ===
def compute_fps(df):
    timestamps = df['Time'].values
    diffs = np.diff(timestamps)
    fps = 1 / diffs
    return fps

fps_head = compute_fps(df_head)
fps_april = compute_fps(df_april)

# === Pozisyon Kararlılığı (Std Sapma) ===
def position_stability(df):
    tx_std = np.std(df['Tx'])
    ty_std = np.std(df['Ty'])
    tz_std = np.std(df['Tz'])
    return tx_std, ty_std, tz_std

stability_head = position_stability(df_head)
stability_april = position_stability(df_april)

# === Decision Margin Karşılaştırması ===
dm_head = df_head['DecisionMargin']
dm_april = df_april['DecisionMargin']

# === Grafikler ===
plt.figure(figsize=(12, 6))

# FPS
plt.subplot(2, 2, 1)
plt.plot(fps_head, label='HeadPose FPS', alpha=0.7)
plt.plot(fps_april, label='AprilTag FPS', alpha=0.7)
plt.title("FPS Karşılaştırması")
plt.xlabel("Frame")
plt.ylabel("FPS")
plt.legend()

# Z Pozisyonu
plt.subplot(2, 2, 2)
plt.plot(df_head['Tz'], label='HeadPose Tz', alpha=0.7)
plt.plot(df_april['Tz'], label='AprilTag Tz', alpha=0.7)
plt.title("Z Ekseni Pozisyonu")
plt.xlabel("Frame")
plt.ylabel("Tz (metre)")
plt.legend()

# Decision Margin
plt.subplot(2, 2, 3)
plt.plot(dm_head, label='HeadPose DM', alpha=0.7)
plt.plot(dm_april, label='AprilTag DM', alpha=0.7)
plt.title("Decision Margin Karşılaştırması")
plt.xlabel("Frame")
plt.ylabel("Decision Margin")
plt.legend()

# Kararlılık Bar Chart
plt.subplot(2, 2, 4)
labels = ['Tx', 'Ty', 'Tz']
x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, stability_head, width, label='HeadPose')
plt.bar(x + width/2, stability_april, width, label='AprilTag')
plt.title("Pozisyon Kararlılığı (Std Sapma)")
plt.xticks(x, labels)
plt.ylabel("Std Sapma")
plt.legend()

plt.tight_layout()
plt.show()

 
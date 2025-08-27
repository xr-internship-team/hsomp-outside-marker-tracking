import pandas as pd
import os
import matplotlib.pyplot as plt

def calculate_angle_range(min_angle, max_angle):
    """Açı aralığını wrap-around (döngü) durumuna göre hesaplar."""
    normal_range = max_angle - min_angle
    if normal_range > 180:  # wrap-around var
        return (180 - abs(min_angle)) + (180 - abs(max_angle))
    else:
        return normal_range

def get_supported_range(log_file, system_name, dm_threshold=1):
    if not os.path.exists(log_file):
        print(f"❌ {system_name} log not found: {log_file}")
        return None

    df = pd.read_csv(log_file)

    required_cols = ["DecisionMargin", "BetaX", "BetaY", "BetaZ"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ {system_name} log missing columns: {missing_cols}")
        return None

    reliable = df[df["DecisionMargin"] > dm_threshold]

    results = {"system": system_name, "threshold": dm_threshold, "axes": {}}

    print(f"\n🔍 {system_name} — Supported Head Rotation Range (DM > {dm_threshold})")
    for axis in ["BetaX", "BetaY", "BetaZ"]:
        min_angle = reliable[axis].min()
        max_angle = reliable[axis].max()
        full_range = calculate_angle_range(min_angle, max_angle)
        results["axes"][axis] = {
            "min": round(min_angle, 2),
            "max": round(max_angle, 2),
            "range": round(full_range, 2)
        }
        print(f"  {axis}: {round(min_angle,1)}° to {round(max_angle,1)}° → Range: {round(full_range,1)}°")

    unstable = df[df["DecisionMargin"] <= dm_threshold]
    if not unstable.empty:
        print(f"\n⚠️ {system_name} starts losing track below DM={dm_threshold}:")
        for axis in ["BetaX", "BetaY", "BetaZ"]:
            min_fail = unstable[axis].min()
            max_fail = unstable[axis].max()
            print(f"  {axis} failure zone: {round(min_fail,1)}° to {round(max_fail,1)}°")

    return results

# Analiz çalıştır
headpose_results = get_supported_range("headpose_log.csv", "HeadPose")
apriltag_results = get_supported_range("apriltag_log.csv", "AprilTag")

# Karşılaştırma grafiği
if headpose_results and apriltag_results:
    axes = ["BetaX", "BetaY", "BetaZ"]
    for axis in axes:
        plt.figure(figsize=(6,4))
        plt.bar(["HeadPose", "AprilTag"],
                [headpose_results["axes"][axis]["range"], apriltag_results["axes"][axis]["range"]],
                color=["blue", "green"])
        plt.title(f"Supported Range for {axis}")
        plt.ylabel("Degrees")
        plt.show()

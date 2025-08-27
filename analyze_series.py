import pandas as pd
import matplotlib.pyplot as plt
import os

test_dirs = ["distance_0.5m", "distance_1.0m", "distance_1.5m", "distance_2.0m"]

for test in test_dirs:
    headpose_path = os.path.join("tests", test, "headpose_log.csv")
    apriltag_path = os.path.join("tests", test, "apriltag_log.csv")

    df_head = pd.read_csv(headpose_path)
    df_april = pd.read_csv(apriltag_path)

    fps_head = 1 / pd.Series(df_head["Time"]).diff()
    fps_april = 1 / pd.Series(df_april["Time"]).diff()

    dm_head = df_head["DecisionMargin"]
    dm_april = df_april["DecisionMargin"]

    plt.figure(figsize=(10, 4))
    plt.plot(fps_head, label="HeadPose FPS", alpha=0.7)
    plt.plot(fps_april, label="AprilTag FPS", alpha=0.7)
    plt.title(f"{test} - FPS Karşılaştırması")
    plt.xlabel("Frame")
    plt.ylabel("FPS")
    plt.legend()
    plt.tight_layout()
    plt.show()


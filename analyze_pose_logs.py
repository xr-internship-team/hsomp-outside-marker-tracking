# analyze_pose_logs.py
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ------------------------
# Yardımcı fonksiyonlar
# ------------------------
def rotation_matrix_from_row(row):
    return np.array([[row['R00'], row['R01'], row['R02']],
                     [row['R10'], row['R11'], row['R12']],
                     [row['R20'], row['R21'], row['R22']]], dtype=float)

def angular_diff_deg(Ra, Rb):
    dR = Ra.T @ Rb
    tr = np.clip(np.trace(dR), -1.0, 3.0)
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))

def compute_fps_stats(t):
    if len(t) < 2: return {}
    dt = np.diff(t)
    fps = 1.0 / dt
    return {
        "fps_med": np.median(fps),
        "fps_min": np.min(fps),
        "fps_max": np.max(fps),
        "jitter_ms": np.std(dt) * 1000
    }

def orientation_jitter(R_list):
    if len(R_list) < 2: return {}
    diffs = [angular_diff_deg(R_list[i-1], R_list[i]) for i in range(1, len(R_list))]
    return {"ori_jitter_rms": np.sqrt(np.mean(np.square(diffs))),
            "ori_jitter_p95": np.percentile(diffs, 95)}

def translation_jitter(pos):
    if len(pos) < 2: return {}
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return {"trans_jitter_mm_rms": np.sqrt(np.mean(np.square(dp)))*1000,
            "trans_jitter_p95": np.percentile(dp,95)*1000}

# ------------------------
# Veri okuma & metrikler
# ------------------------
def load_log(path, algo):
    df = pd.read_csv(path)
    df['algo'] = algo
    df = df.sort_values('Time')
    return df

def compute_metrics(df):
    out = []
    for (algo, tag), g in df.groupby(['algo','ID']):
        t = g['Time'].to_numpy()
        pos = g[['Tx','Ty','Tz']].to_numpy(float)
        R_list = [rotation_matrix_from_row(row) for _, row in g.iterrows()]
        row = {"algo": algo, "ID": tag}
        row.update(compute_fps_stats(t))
        row.update(orientation_jitter(R_list))
        row.update(translation_jitter(pos))
        if 'DecisionMargin' in g.columns:
            dm = g['DecisionMargin'].dropna().to_numpy()
            if len(dm):
                row.update({"dm_mean": np.mean(dm),
                            "dm_p95": np.percentile(dm,95)})
        out.append(row)
    return pd.DataFrame(out)

# ------------------------
# Grafik fonksiyonları
# ------------------------
def plot_hist(data, title, xlabel, path):
    if len(data) == 0: return
    plt.figure()
    plt.hist(data, bins=50)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_timeseries(t, y, title, ylabel, path):
    if len(y) == 0: return
    plt.figure()
    plt.plot(t, y)
    plt.title(title); plt.xlabel("time (s)"); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(path); plt.close()

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facemesh", type=str)
    ap.add_argument("--apriltag", type=str)
    ap.add_argument("--out", type=str, default="pose_report")
    args = ap.parse_args()

    outdir = Path(args.out); (outdir/"figs").mkdir(parents=True, exist_ok=True)

    dfs = []
    if args.facemesh: dfs.append(load_log(Path(args.facemesh),"FaceMesh"))
    if args.apriltag: dfs.append(load_log(Path(args.apriltag),"AprilTag"))
    df = pd.concat(dfs, ignore_index=True)

    metrics = compute_metrics(df)
    metrics.to_csv(outdir/"summary_metrics.csv", index=False)

    # Grafikler
    for algo, g in df.groupby("algo"):
        t = g['Time'].to_numpy()
        plot_hist(np.diff(t)*1000, f"{algo} Δt", "Δt (ms)", outdir/"figs"/f"{algo}_dt_hist.png")
        if 'DecisionMargin' in g.columns:
            plot_timeseries(t, g['DecisionMargin'], f"{algo} DecisionMargin", "DM", outdir/"figs"/f"{algo}_dm.png")

        R_list = [rotation_matrix_from_row(r) for _, r in g.iterrows()]
        if len(R_list) > 1:
            ang = [angular_diff_deg(R_list[i-1],R_list[i]) for i in range(1,len(R_list))]
            plot_timeseries(t[1:], ang, f"{algo} Δangle", "deg", outdir/"figs"/f"{algo}_angle.png")

        pos = g[['Tx','Ty','Tz']].to_numpy(float)
        if len(pos) > 1:
            dp = np.linalg.norm(np.diff(pos,axis=0),axis=1)*1000
            plot_timeseries(t[1:], dp, f"{algo} Δpos", "mm", outdir/"figs"/f"{algo}_pos.png")

    print("Metrikler kaydedildi:", outdir)

if __name__=="__main__":
    main()

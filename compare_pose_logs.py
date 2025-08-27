# compare_pose_logs.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def rotation_matrix_from_row(row):
    return np.array([[row['R00'], row['R01'], row['R02']],
                     [row['R10'], row['R11'], row['R12']],
                     [row['R20'], row['R21'], row['R22']]], dtype=float)

def angular_diff_deg(Ra, Rb):
    dR = Ra.T @ Rb
    tr = np.clip(np.trace(dR), -1.0, 3.0)
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))

def load_log(path, algo):
    df = pd.read_csv(path).sort_values("Time")
    df["algo"] = algo
    return df

def compute_metrics(df):
    metrics = []
    for algo, g in df.groupby("algo"):
        t = g["Time"].to_numpy()
        dt = np.diff(t)
        fps = 1/np.median(dt) if len(dt)>0 else np.nan
        jitter = np.std(dt)*1000 if len(dt)>0 else np.nan

        pos = g[["Tx","Ty","Tz"]].to_numpy(float)
        trans = np.linalg.norm(np.diff(pos,axis=0),axis=1)*1000 if len(pos)>1 else []

        R_list = [rotation_matrix_from_row(r) for _,r in g.iterrows()]
        ang = [angular_diff_deg(R_list[i-1],R_list[i]) for i in range(1,len(R_list))] if len(R_list)>1 else []

        dm = g["DecisionMargin"].dropna().to_numpy() if "DecisionMargin" in g else []

        metrics.append({
            "algo": algo,
            "fps": fps,
            "jitter_ms": jitter,
            "trans_jitter_mm_rms": np.sqrt(np.mean(np.square(trans))) if len(trans) else np.nan,
            "ang_jitter_deg_rms": np.sqrt(np.mean(np.square(ang))) if len(ang) else np.nan,
            "dm_mean": np.mean(dm) if len(dm) else np.nan
        })
    return pd.DataFrame(metrics)

def plot_comparison(metrics, outdir):
    outdir.mkdir(exist_ok=True, parents=True)
    for col in ["fps","jitter_ms","trans_jitter_mm_rms","ang_jitter_deg_rms","dm_mean"]:
        plt.figure()
        plt.bar(metrics["algo"], metrics[col])
        plt.title(f"{col} comparison")
        plt.ylabel(col)
        plt.savefig(outdir/f"{col}_compare.png")
        plt.close()

def auto_analysis(metrics):
    print("\n=== ANALYSIS ===")
    algos = metrics["algo"].tolist()
    if len(algos)<2:
        print("Tek algo var, karşılaştırma yapılamaz.")
        return
    a, b = algos
    row_a, row_b = metrics.iloc[0], metrics.iloc[1]

    def better(metric, lower_better=False):
        va, vb = row_a[metric], row_b[metric]
        if np.isnan(va) or np.isnan(vb): return "n/a"
        if lower_better:
            winner = a if va<vb else b
        else:
            winner = a if va>vb else b
        return f"{winner} better ({va:.2f} vs {vb:.2f})"

    print("FPS:", better("fps"))
    print("Frame jitter (ms):", better("jitter_ms", lower_better=True))
    print("Translation jitter (mm):", better("trans_jitter_mm_rms", lower_better=True))
    print("Orientation jitter (deg):", better("ang_jitter_deg_rms", lower_better=True))
    print("DecisionMargin:", better("dm_mean"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facemesh", type=str)
    ap.add_argument("--apriltag", type=str)
    ap.add_argument("--out", type=str, default="pose_figs")
    args = ap.parse_args()

    dfs=[]
    if args.facemesh: dfs.append(load_log(Path(args.facemesh),"FaceMesh"))
    if args.apriltag: dfs.append(load_log(Path(args.apriltag),"AprilTag"))
    df=pd.concat(dfs,ignore_index=True)

    metrics=compute_metrics(df)
    print("\n=== SUMMARY METRICS ===")
    print(metrics.round(3).to_string(index=False))

    plot_comparison(metrics, Path(args.out))
    auto_analysis(metrics)
    print(f"\nGrafikler kaydedildi: {args.out}")

if __name__=="__main__":
    main()

# pose_dashboard.py
# -*- coding: utf-8 -*-
"""
Generate a full HTML dashboard (tables + charts + auto-insights) comparing FaceMesh vs AprilTag logs.

Usage:
    python pose_dashboard.py --facemesh headpose_log.csv --apriltag apriltag_log.csv
Options:
    --out pose_report     : output directory
    --gap-ms 200          : gap threshold to split segments and compute relock stats

Input CSV columns (as per your logger):
['Time','ID','Tx','Ty','Tz',
 'R00','R01','R02','R10','R11','R12','R20','R21','R22',
 'BetaX','BetaY','BetaZ','AlphaX','AlphaY','AlphaZ','DecisionMargin','Confidence','RScale']
Only the Time, ID, Tx/Ty/Tz, R** columns are strictly required. Others are optional.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import os
import json
import datetime

# ----------------------
# Utilities
# ----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def rotation_matrix_from_row(row) -> np.ndarray:
    return np.array([[row['R00'], row['R01'], row['R02']],
                     [row['R10'], row['R11'], row['R12']],
                     [row['R20'], row['R21'], row['R22']]], dtype=float)

def angular_diff_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra.T @ Rb
    tr = np.clip(np.trace(dR), -1.0, 3.0)
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))

def _quat_average(rot_mats_subset):
    if len(rot_mats_subset) == 0:
        return np.eye(3)
    Q = []
    for M in rot_mats_subset:
        try:
            Q.append(R.from_matrix(M).as_quat())
        except Exception:
            Q.append(np.array([0.0,0.0,0.0,1.0], dtype=float))
    Q = np.vstack(Q)
    q_sum = Q.sum(axis=0)
    nrm = np.linalg.norm(q_sum)
    if nrm < 1e-12:
        return np.eye(3)
    q_mean = q_sum / nrm
    return R.from_quat(q_mean).as_matrix()

# ----------------------
# Loading
# ----------------------
def load_log(path: Path, algo: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = ['Time','ID','Tx','Ty','Tz','R00','R01','R02','R10','R11','R12','R20','R21','R22']
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"{path} missing columns: {miss}")
    df = df.copy()
    df['algo'] = algo
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time')
    return df

# ----------------------
# Metrics
# ----------------------
def compute_fps_stats(t: np.ndarray) -> dict:
    if len(t) < 2: 
        return {"fps_med": np.nan, "fps_min": np.nan, "fps_max": np.nan, "jitter_ms": np.nan, "dt_ms_p95": np.nan}
    dt = np.diff(t)
    fps = 1.0 / dt
    return {
        "fps_med": float(np.median(fps)),
        "fps_min": float(np.min(fps)),
        "fps_max": float(np.max(fps)),
        "jitter_ms": float(np.std(dt) * 1000.0),
        "dt_ms_p95": float(np.percentile(dt, 95) * 1000.0),
    }

def segment_stats(t: np.ndarray, gap_thresh_s: float) -> dict:
    if len(t) == 0:
        return {"n_segments": 0, "mean_seg_s": np.nan, "p95_gap_ms": np.nan, "n_gaps": 0, "relock_p50_ms": np.nan}
    starts = [0]
    for i in range(1,len(t)):
        if t[i] - t[i-1] > gap_thresh_s:
            starts.append(i)
    segs=[]
    for s_idx, e_idx in zip(starts, starts[1:]+[len(t)]):
        dur = t[e_idx-1] - t[s_idx]
        segs.append((s_idx, e_idx-1, dur))
    gap_starts = [segs[i][1] for i in range(len(segs)-1)]
    gap_ends   = [segs[i+1][0] for i in range(len(segs)-1)]
    gaps = [t[g_end] - t[g_start] for g_start, g_end in zip(gap_starts, gap_ends)]
    return {
        "n_segments": len(segs),
        "mean_seg_s": float(np.mean([s[2] for s in segs])) if segs else np.nan,
        "p95_gap_ms": float(np.percentile(gaps,95)*1000.0) if gaps else 0.0,
        "n_gaps": len(gaps),
        "relock_p50_ms": float(np.median(gaps)*1000.0) if gaps else 0.0,
    }

def orientation_jitter_deg(R_list: list) -> dict:
    if len(R_list) < 2:
        return {"ang_jitter_deg_rms": np.nan, "ang_deg_p95": np.nan}
    d = [angular_diff_deg(R_list[i-1], R_list[i]) for i in range(1,len(R_list))]
    return {
        "ang_jitter_deg_rms": float(np.sqrt(np.mean(np.square(d)))),
        "ang_deg_p95": float(np.percentile(d,95))
    }

def position_stability(pos: np.ndarray) -> dict:
    if len(pos) < 2:
        return {"trans_jitter_mm_rms": np.nan, "trans_mm_p95": np.nan}
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return {
        "trans_jitter_mm_rms": float(np.sqrt(np.mean(np.square(dp))) * 1000.0),
        "trans_mm_p95": float(np.percentile(dp,95) * 1000.0)
    }

def drift_over_time(pos: np.ndarray, R_list: list, frac: float=0.1) -> dict:
    n = len(pos)
    if n < 10 or len(R_list) < 2:
        return {"drift_pos_mm": np.nan, "drift_ang_deg": np.nan}
    k = max(1, int(n*frac))
    p_start = pos[:k].mean(axis=0)
    p_end   = pos[-k:].mean(axis=0)
    drift_pos = float(np.linalg.norm(p_end - p_start) * 1000.0)
    R_start = _quat_average(R_list[:k])
    R_end   = _quat_average(R_list[-k:])
    drift_ang = float(angular_diff_deg(R_start, R_end))
    return {"drift_pos_mm": drift_pos, "drift_ang_deg": drift_ang}

def compute_metrics_per_group(df: pd.DataFrame, gap_s: float) -> pd.DataFrame:
    rows=[]
    for (algo, tag_id), g in df.groupby(['algo','ID']):
        t = g['Time'].to_numpy()
        pos = g[['Tx','Ty','Tz']].to_numpy(dtype=float)
        R_list = [rotation_matrix_from_row(row) for _, row in g.iterrows()]
        row = {"algo": algo, "ID": tag_id}
        row.update(compute_fps_stats(t))
        row.update(segment_stats(t, gap_s))
        row.update(orientation_jitter_deg(R_list))
        row.update(position_stability(pos))
        row.update(drift_over_time(pos, R_list))
        if 'DecisionMargin' in g.columns:
            dm = pd.to_numeric(g['DecisionMargin'], errors='coerce').dropna().to_numpy()
            if len(dm):
                row.update({"dm_mean": float(np.mean(dm)),
                            "dm_p05": float(np.percentile(dm,5)),
                            "dm_p95": float(np.percentile(dm,95))})
        rows.append(row)
    return pd.DataFrame(rows)

def aggregate_per_algo(df: pd.DataFrame, gap_s: float) -> pd.DataFrame:
    m = compute_metrics_per_group(df, gap_s)
    if m.empty: 
        return m
    agg = []
    for algo, g in m.groupby('algo'):
        agg.append({
            "algo": algo,
            "fps_med": float(np.nanmean(g['fps_med'])),
            "jitter_ms": float(np.nanmean(g['jitter_ms'])),
            "dt_ms_p95": float(np.nanmean(g['dt_ms_p95'])),
            "ang_jitter_deg_rms": float(np.nanmean(g['ang_jitter_deg_rms'])),
            "trans_jitter_mm_rms": float(np.nanmean(g['trans_jitter_mm_rms'])),
            "relock_p50_ms": float(np.nanmean(g['relock_p50_ms'])),
            "p95_gap_ms": float(np.nanmean(g['p95_gap_ms'])),
            "drift_pos_mm": float(np.nanmean(g['drift_pos_mm'])),
            "drift_ang_deg": float(np.nanmean(g['drift_ang_deg'])),
            "dm_mean": float(np.nanmean(g['dm_mean'])) if 'dm_mean' in g else np.nan
        })
    return pd.DataFrame(agg)

# ----------------------
# Plotting
# ----------------------
def plot_hist(data, title, xlabel, out_path: Path):
    if len(data) == 0: return None
    plt.figure()
    plt.hist(data, bins=50)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path

def plot_line(x, y, title, xlabel, ylabel, out_path: Path):
    if len(y) == 0: return None
    plt.figure()
    plt.plot(x, y)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path

def generate_figures(df: pd.DataFrame, figs_dir: Path) -> list:
    paths=[]
    for algo, g in df.groupby('algo'):
        t = g['Time'].to_numpy()
        if len(t) > 1:
            dt_ms = np.diff(t) * 1000.0
            p = plot_hist(dt_ms, f"{algo} inter-frame Δt", "Δt (ms)", figs_dir / f"{algo}_dt_hist.png")
            if p: paths.append(p)
            p = plot_line(t[1:], dt_ms, f"{algo} Δt over time", "time (s)", "Δt (ms)", figs_dir / f"{algo}_dt_series.png")
            if p: paths.append(p)
        if 'DecisionMargin' in g.columns:
            dm = pd.to_numeric(g['DecisionMargin'], errors='coerce').dropna().to_numpy()
            p = plot_hist(dm, f"{algo} DecisionMargin", "DM", figs_dir / f"{algo}_dm_hist.png")
            if p: paths.append(p)
            p = plot_line(t[:len(dm)], dm, f"{algo} DecisionMargin over time", "time (s)", "DM", figs_dir / f"{algo}_dm_series.png")
            if p: paths.append(p)
        R_list = [rotation_matrix_from_row(row) for _, row in g.iterrows()]
        if len(R_list) > 1:
            ang = [angular_diff_deg(R_list[i-1], R_list[i]) for i in range(1, len(R_list))]
            p = plot_hist(np.array(ang), f"{algo} Δangle", "deg", figs_dir / f"{algo}_ang_hist.png")
            if p: paths.append(p)
            p = plot_line(t[1:], ang, f"{algo} Δangle over time", "time (s)", "deg", figs_dir / f"{algo}_ang_series.png")
            if p: paths.append(p)
        pos = g[['Tx','Ty','Tz']].to_numpy(dtype=float)
        if len(pos) > 1:
            dp = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0
            p = plot_hist(dp, f"{algo} Δpos", "mm", figs_dir / f"{algo}_dpos_hist.png")
            if p: paths.append(p)
            p = plot_line(t[1:], dp, f"{algo} Δpos over time", "time (s)", "mm", figs_dir / f"{algo}_dpos_series.png")
            if p: paths.append(p)
    return paths

def bar_compare(agg_df: pd.DataFrame, metric: str, out_path: Path, title: str, ylabel: str):
    if agg_df.empty or metric not in agg_df.columns: return None
    plt.figure()
    plt.bar(agg_df['algo'], agg_df[metric])
    plt.title(title); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_path); plt.close()
    return out_path

# ----------------------
# Auto insights
# ----------------------
def who_is_better(agg_df: pd.DataFrame, metric: str, lower_better: bool) -> str:
    if agg_df.shape[0] < 2 or metric not in agg_df.columns:
        return "n/a"
    a0 = agg_df.iloc[0]; a1 = agg_df.iloc[1]
    v0 = a0[metric]; v1 = a1[metric]
    if pd.isna(v0) or pd.isna(v1): return "n/a"
    if lower_better:
        winner = a0['algo'] if v0 < v1 else a1['algo']
        diff = (min(v0, v1) / max(v0, v1 + 1e-12) - 1.0) * 100.0
    else:
        winner = a0['algo'] if v0 > v1 else a1['algo']
        diff = (max(v0, v1) / (min(v0, v1) + 1e-12) - 1.0) * 100.0
    return f"{winner} (≈{abs(diff):.1f}%)"

def insight_paragraphs(agg_df: pd.DataFrame) -> list:
    items=[]
    items.append(f"- **FPS (medyan)**: {who_is_better(agg_df, 'fps_med', lower_better=False)} daha yüksek.")
    items.append(f"- **Frame jitter (std, ms)**: {who_is_better(agg_df, 'jitter_ms', lower_better=True)} daha stabil.")
    items.append(f"- **Δt 95p (ms)**: {who_is_better(agg_df, 'dt_ms_p95', lower_better=True)} daha iyi p95 aralığı.")
    items.append(f"- **Orientasyon jitter (RMS, °)**: {who_is_better(agg_df, 'ang_jitter_deg_rms', lower_better=True)} daha düşük açısal gürültü.")
    items.append(f"- **Translasyon jitter (RMS, mm)**: {who_is_better(agg_df, 'trans_jitter_mm_rms', lower_better=True)} daha düşük konumsal gürültü.")
    items.append(f"- **Relock medyan (ms)**: {who_is_better(agg_df, 'relock_p50_ms', lower_better=True)} daha hızlı yeniden yakalıyor.")
    items.append(f"- **Drift (mm, °)**: {who_is_better(agg_df, 'drift_pos_mm', lower_better=True)} / {who_is_better(agg_df, 'drift_ang_deg', lower_better=True)} daha az sürüklenme.")
    if 'dm_mean' in agg_df.columns:
        items.append(f"- **DecisionMargin ort.**: {who_is_better(agg_df, 'dm_mean', lower_better=False)} daha yüksek güven.")
    return items

# ----------------------
# HTML report
# ----------------------
METRIC_DOC = """
<h3>Metrix Sözlüğü</h3>
<ul>
<li><b>FPS (medyan)</b>: Sistemin sürdürülebilir kare işleme hızı. Yüksek olması akıcılık açısından iyidir.</li>
<li><b>Frame jitter (ms)</b>: Kareler arası sürelerin standart sapması. Düşük olması stabil zamanlama demektir.</li>
<li><b>Δt p95 (ms)</b>: Kareler arası sürenin 95. yüzdelik değeri. Kuyruktaki gecikmeleri temsil eder; düşük olması iyidir.</li>
<li><b>Orientation jitter (RMS, °)</b>: Ardışık pozlar arasındaki açısal değişimin RMS’i. Düşükse rotasyon gürültüsü azdır.</li>
<li><b>Translation jitter (RMS, mm)</b>: Ardışık pozlar arasındaki konum değişiminin RMS’i. Düşükse poz gürültüsü azdır.</li>
<li><b>Relock median (ms)</b>: Takip kaybı sonrası yeniden yakalama sürelerinin medyanı. Düşükse daha hızlı toparlıyor.</li>
<li><b>p95 gap (ms)</b>: Takip kesintilerinin 95. yüzdelik süresi. Düşükse uzun kopmalar azdır.</li>
<li><b>Drift (mm, °)</b>: Sekans başı ve sonu ortalamaları arasındaki fark (konum ve açı). Düşükse uzun süre stabil demektir.</li>
<li><b>DecisionMargin</b>: Algoritmanın iç güven puanı (ölçeksiz). Yüksekse daha emin demektir.</li>
</ul>
"""

def df_to_html(df: pd.DataFrame) -> str:
    return df.round(3).to_html(index=False, border=0, classes="table")

def html_report(out_dir: Path, per_group: pd.DataFrame, agg_df: pd.DataFrame, figs: list[Path]) -> Path:
    html_path = out_dir / "reportstable.html"
    when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    imgs_html = "\n".join([f'<div class="img"><img src="figs/{p.name}" alt="{p.name}"/></div>' for p in figs])
    insights = "\n".join([f"<li>{s}</li>" for s in insight_paragraphs(agg_df)])
    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:24px}
    h1,h2,h3{margin:8px 0}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .section{margin:22px 0}
    .table{border-collapse:collapse;width:100%}
    .table th,.table td{border:1px solid #ddd;padding:6px 8px}
    .table thead{background:#f2f2f2}
    .img{margin:10px 0}
    img{max-width:100%;height:auto;border:1px solid #eee;padding:4px}
    .callout{background:#f9f9f9;border-left:4px solid #aaa;padding:8px 12px;margin:10px 0}
    nav a{margin-right:12px}
    """
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pose Dashboard</title>
<style>{css}</style></head>
<body>
<nav>
  <a href="#overview">Özet</a>
  <a href="#insights">Çıkarımlar</a>
  <a href="#tables">Tablolar</a>
  <a href="#figs">Grafikler</a>
  <a href="#docs">Metrix Sözlüğü</a>
</nav>

<h1 id="overview">Pose Analiz Raporu</h1>
<div class="callout">Oluşturulma: {when} — Klasör: <code>{out_dir}</code></div>

<h2 id="insights">Kısa Çıkarımlar</h2>
<ul>
{insights}
</ul>

<h2 id="tables">Tablolar</h2>
<div class="section">
  <h3>Algoritma Bazında Özet</h3>
  {df_to_html(agg_df)}
</div>
<div class="section">
  <h3>(Algo, ID) Bazında</h3>
  {df_to_html(per_group)}
</div>

<h2 id="figs">Grafikler</h2>
<div class="grid">
{imgs_html}
</div>

<h2 id="docs">Açıklamalar</h2>
<div class="section">{METRIC_DOC}</div>

</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return html_path

# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facemesh", type=str, help="FaceMesh CSV")
    ap.add_argument("--apriltag", type=str, help="AprilTag CSV")
    ap.add_argument("--out", type=str, default="pose_report", help="Output directory")
    ap.add_argument("--gap-ms", type=float, default=200.0, help="Gap threshold (ms)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    figs_dir = out_dir / "figs"
    ensure_dir(figs_dir)

    frames=[]
    if args.facemesh and Path(args.facemesh).exists():
        frames.append(load_log(Path(args.facemesh),"FaceMesh"))
    if args.apriltag and Path(args.apriltag).exists():
        frames.append(load_log(Path(args.apriltag),"AprilTag"))
    if not frames:
        raise SystemExit("No input CSVs found. Provide --facemesh and/or --apriltag")

    df = pd.concat(frames, ignore_index=True)
    gap_s = float(args.gap_ms) / 1000.0

    per_group = compute_metrics_per_group(df, gap_s)
    agg_df = aggregate_per_algo(df, gap_s)

    # Save tables (csv) for reference
    per_group.to_csv(out_dir/"summary_metrics.csv", index=False)
    agg_df.to_csv(out_dir/"summary_metrics_aggregated.csv", index=False)

    # Per-algo timeseries & histograms
    fig_list = generate_figures(df, figs_dir)

    # Comparative bars
    comp_specs = [
        ("fps_med", "Median FPS per algo", "FPS", "algo_fps_med.png"),
        ("jitter_ms", "Frame interval jitter per algo", "jitter (ms)", "algo_jitter.png"),
        ("dt_ms_p95", "Δt p95 per algo", "ms", "algo_dt95.png"),
        ("ang_jitter_deg_rms", "Orientation jitter per algo", "RMS Δangle (deg)", "algo_angjit.png"),
        ("trans_jitter_mm_rms", "Translation jitter per algo", "RMS Δpos (mm)", "algo_transjit.png"),
        ("relock_p50_ms", "Relock median per algo", "ms", "algo_relock.png"),
        ("p95_gap_ms", "Gap p95 per algo", "ms", "algo_gap95.png"),
        ("drift_pos_mm", "Drift (position) per algo", "mm", "algo_drift_pos.png"),
        ("drift_ang_deg", "Drift (angle) per algo", "deg", "algo_drift_ang.png"),
        ("dm_mean", "DecisionMargin mean per algo", "DM", "algo_dm_mean.png"),
    ]
    for metric, title, ylabel, fname in comp_specs:
        p = bar_compare(agg_df, metric, figs_dir / fname, title, ylabel)
        if p: fig_list.append(p)

    # HTML report
    report_path = html_report(out_dir, per_group, agg_df, fig_list)

    # Minimal console info
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main()

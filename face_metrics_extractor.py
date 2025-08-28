# face_metrics_extractor.py
# -*- coding: utf-8 -*-
import argparse, csv, time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# =========================
# Parametreler (CLI ile de değiştirilebilir)
# =========================
DEF_CAM_INDEX   = 0
DEF_TAG_SIZE_M  = 0.08      # AprilTag kenar uzunluğu (metre)
DEF_CALIB_PATH  = "calib_params.npz"
DEF_OUT_CSV     = "face_metrics_log.csv"
EMA_ALPHA       = 0.15      # EMA yumuşatma (0.1–0.3 arası iyi)
APRIL_FAMILY    = "tag36h11"  # kullandığın aileyi buraya yaz

# -------------------------
# FaceMesh / PnP ayarları
# -------------------------
# solvePnP'te kullanılacak noktalar (MediaPipe FaceMesh indeksleri)
IDX_PNP = [1, 152, 162, 389, 61, 291]

# Bu 6 noktanın 3B model koordinatları (metre).
# Sıra IDX_PNP ile birebir aynı olmalı.
MODEL_POINTS_BASE = np.array([
    ( 0.000,  0.000,  0.000),   # 1   - Burun ucu
    ( 0.000, -0.095, -0.010),   # 152 - Çene
    (-0.065,  0.045, -0.055),   # 162 - Sol temporal
    ( 0.065,  0.045, -0.055),   # 389 - Sağ temporal
    (-0.032, -0.040, -0.020),   # 61  - Sol ağız köşesi
    ( 0.032, -0.040, -0.020)    # 291 - Sağ ağız köşesi
], dtype=np.float64)

# Düzleme yakın kabul edilen ve k tahmini için kullanılacak çiftler (ad, (mp_i, mp_j), ağırlık)
SCALE_PAIRS = [
    ("agiz_genisligi",  (61, 291), 1.0),
    ("sakak_genisligi", (162, 389), 1.0),
    # ("burun_cene", (1, 152), 0.2),  # istersen düşük ağırlıkla ekle
]

# Rapor/CSV’ye yazılacak metrik çiftleri
REPORT_PAIRS = {
    "agiz_genisligi":  (61, 291),
    "sakak_genisligi": (162, 389),
    "burun_cene":      (1, 152),
    # ör. "gozler_arasi": (33, 263)  # dilersen ekleyebilirsin
}

# Ekranda kolay kontrol için bazı önemli noktalar
FACE_IDS_DRAW = [1, 33, 263, 61, 291, 152, 162, 389]

# =========================
# Yardımcı fonksiyonlar
# =========================
def ema(prev, new, alpha):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def detect_apriltag_scale(gray, tag_size_m, detector: Detector):
    """
    pupil_apriltags ile AprilTag tespit et, ortalama kenar uzunluğundan m/px ölçeğini döndür.
    """
    # detector.detect: grayscale img, returns list of detections
    # each has .corners (4x2), .center (2,), .tag_id, .decision_margin
    detections = detector.detect(
        gray,
        estimate_tag_pose=False  # sadece ölçek için köşeler yeterli
    )
    if not detections:
        return None, None
    det = max(detections, key=lambda d: d.decision_margin)
    corners = np.array(det.corners, dtype=np.float32)  # [tl,tr,br,bl]
    sides = [
        np.linalg.norm(corners[1]-corners[0]),
        np.linalg.norm(corners[2]-corners[1]),
        np.linalg.norm(corners[3]-corners[2]),
        np.linalg.norm(corners[0]-corners[3]),
    ]
    p = float(np.mean(sides))
    if p <= 1e-6:
        return None, None
    return float(tag_size_m / p), corners

def isotropic_k_from_pairs(model_pts, idx_pnp, lm_px, m_per_px, scale_pairs):
    """
    Düzlemsel kabul edilen mesafelerden (2B px -> m) LS ile tek ölçek k hesaplar.
    model_pts: solvePnP model noktaları (k=1 hâli), sırası idx_pnp ile aynı.
    scale_pairs: list of (name, (mp_i, mp_j), weight)
    """
    mp_to_model = {mp_idx: k for k, mp_idx in enumerate(idx_pnp)}
    num = 0.0
    den = 0.0
    for _, (mp_i, mp_j), w in scale_pairs:
        if mp_i not in mp_to_model or mp_j not in mp_to_model:
            continue
        mi = mp_to_model[mp_i]
        mj = mp_to_model[mp_j]
        d3 = np.linalg.norm(model_pts[mi] - model_pts[mj])  # k=1
        if d3 < 1e-9:
            continue
        d_px = np.linalg.norm(lm_px[mp_i] - lm_px[mp_j])
        d_m  = d_px * m_per_px
        num += w * d3 * d_m
        den += w * (d3 ** 2)
    if den < 1e-9:
        return 1.0
    return float(num / den)

def draw_points(frame, lm_px, ids, color=(0,255,255)):
    for idx in ids:
        if 0 <= idx < len(lm_px):
            u, v = int(lm_px[idx][0]), int(lm_px[idx][1])
            cv2.circle(frame, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(idx), (u+3, v-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

# =========================
# Ana akış
# =========================
def main():
    ap = argparse.ArgumentParser(description="AprilTag (pupil_apriltags) ölçekli yüz metrikleri çıkarıcı")
    ap.add_argument("--camera", type=int, default=DEF_CAM_INDEX)
    ap.add_argument("--tag-size", type=float, default=DEF_TAG_SIZE_M, help="AprilTag kenarı (metre)")
    ap.add_argument("--calib", type=str, default=DEF_CALIB_PATH, help="calib_params.npz yolu")
    ap.add_argument("--out", type=str, default=DEF_OUT_CSV, help="çıktı CSV dosyası")
    ap.add_argument("--family", type=str, default=APRIL_FAMILY, help="AprilTag ailesi (örn. tag36h11)")
    ap.add_argument("--display", action="store_true", help="canlı önizleme penceresi göster")
    args = ap.parse_args()

    # Kamera kalibrasyonu
    with np.load(args.calib) as data:
        K = data["camera_matrix"]
        D = data["dist_coeffs"]

    # Video
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    # FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # AprilTag Detector
    at_detector = Detector(
        families=args.family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )

    # CSV
    csv_file = open(args.out, "w", newline="")
    writer = csv.writer(csv_file)
    header = [
        "timestamp",
        "m_per_px", "k_personal",
        "agiz_genisligi(m)", "sakak_genisligi(m)", "burun_cene(m)",
        # PnP çıktıları:
        "Tx","Ty","Tz",
        "R00","R01","R02","R10","R11","R12","R20","R21","R22",
        "reproj_rmse"
    ]
    writer.writerow(header)

    # EMA & istatistik
    mpp_ema = None
    k_ema   = 1.0
    mpp_hist = deque(maxlen=100)
    k_hist   = deque(maxlen=100)

    last_stats_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            # --- AprilTag ölçeği ---
            mpp, tag_corners = detect_apriltag_scale(gray, args.tag_size, at_detector)
            if mpp is not None:
                mpp_ema = ema(mpp_ema, mpp, EMA_ALPHA)
                mpp_hist.append(mpp)

            # --- FaceMesh ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = face_mesh.process(rgb)
            if not out.multi_face_landmarks:
                if args.display:
                    cv2.imshow("Face Metrics", frame)
                    if cv2.waitKey(1) == 27:
                        break
                continue

            lm = out.multi_face_landmarks[0].landmark
            lm_px = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

            # Ölçümler: metre (planar kabul)
            measured = {}
            if mpp_ema is not None:
                for name, (i, j) in REPORT_PAIRS.items():
                    d_px = np.linalg.norm(lm_px[i] - lm_px[j])
                    measured[name] = d_px * mpp_ema  # metre

            # Kişiye özel k (izotropik)
            model_pts = MODEL_POINTS_BASE.copy()
            if mpp_ema is not None:
                k_ls = isotropic_k_from_pairs(model_pts, IDX_PNP, lm_px, mpp_ema, SCALE_PAIRS)
                k_ema = ema(k_ema, k_ls, EMA_ALPHA)
                k_hist.append(k_ema)

            model_pts_scaled = model_pts * float(k_ema)

            # PnP girişleri
            image_points = np.array([(lm[i].x * w, lm[i].y * h) for i in IDX_PNP], dtype=np.float64)

            ok_pnp, rvec, tvec = cv2.solvePnP(
                model_pts_scaled, image_points, K, D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            reproj_rmse = -1.0
            R_cv = np.eye(3, dtype=np.float64)
            if ok_pnp:
                R_cv, _ = cv2.Rodrigues(rvec)
                proj, _ = cv2.projectPoints(model_pts_scaled, rvec, tvec, K, D)
                err = np.linalg.norm(image_points - proj.reshape(-1,2), axis=1)
                reproj_rmse = float(np.sqrt(np.mean(err**2)))

            # CSV satırı
            ts = time.time()
            row = [
                ts,
                float(mpp_ema) if mpp_ema is not None else -1.0,
                float(k_ema),
                measured.get("agiz_genisligi", -1.0),
                measured.get("sakak_genisligi", -1.0),
                measured.get("burun_cene", -1.0),
                *(tvec.reshape(3).tolist() if ok_pnp else [-1.0, -1.0, -1.0]),
                *R_cv.reshape(-1).tolist(),
                reproj_rmse
            ]
            writer.writerow(row)

            # Görsel
            if args.display:
                # Landmark göstergeleri
                draw_points(frame, lm_px, FACE_IDS_DRAW)
                # m/px & k overlay
                if mpp_ema is not None:
                    txt = f"k:{k_ema:.3f}  m/px:{mpp_ema*1000:.3f} mm/px"
                    cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    # bazı metrikler
                    if "agiz_genisligi" in measured:
                        cv2.putText(frame, f"Ağız: {measured['agiz_genisligi']*1000:.1f} mm",
                                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                    if "sakak_genisligi" in measured:
                        cv2.putText(frame, f"Sakak: {measured['sakak_genisligi']*1000:.1f} mm",
                                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                    if "burun_cene" in measured:
                        cv2.putText(frame, f"Burun-Çene: {measured['burun_cene']*1000:.1f} mm",
                                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

                if ok_pnp:
                    # eksen çiz
                    axis_len = 0.03
                    cv2.drawFrameAxes(frame, K, D, rvec, tvec, axis_len)
                    cv2.putText(frame, f"RMSE: {reproj_rmse:.2f}px", (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

                # Tag köşelerini çizmek istersen:
                # if tag_corners is not None:
                #     for a,b in zip([0,1,2,3],[1,2,3,0]):
                #         pt1 = tuple(map(int, tag_corners[a]))
                #         pt2 = tuple(map(int, tag_corners[b]))
                #         cv2.line(frame, pt1, pt2, (0,200,0), 2)

                cv2.imshow("Face Metrics", frame)
                if cv2.waitKey(1) == 27:  # ESC
                    break

            # ara ara konsola kısa özet yaz
            now = time.time()
            if now - last_stats_print > 5.0:
                last_stats_print = now
                if len(mpp_hist) > 0:
                    print(f"[INFO] m/px EMA: {mpp_ema:.6f}  | k EMA: {k_ema:.3f}  | RMSE: {reproj_rmse:.2f}")

    finally:
        cap.release()
        face_mesh.close()
        csv_file.close()
        if 'cv2' in globals():
            try:
                cv2.destroyAllWindows()
            except:
                pass

        # Kısa özet
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr)>0 else float('nan')
        print("\n=== Final Stats ===")
        print(f"m/px mean: {safe_mean(mpp_hist):.6f}")
        print(f"k mean   : {safe_mean(k_hist):.4f}")
        print(f"CSV saved to: {args.out}")

if __name__ == "__main__":
    main()

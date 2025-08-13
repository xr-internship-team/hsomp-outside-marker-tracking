import cv2, os, csv, json, time
import numpy as np
from scipy.spatial.transform import Rotation as R
from kalmanFilter import PoseKalmanFilter  # fusion içinde kullanılacak ama import sende dursun

from config import (
    UDP_IP, UDP_PORT, FRONT_TAG_ID, EXTRINSICS_FILE,
    TAG_SIZES, EMA_ALPHA, CSV_PATH, FAMILY
)
from tracker.detector_ptag import PTagDetector
from tracker.transforms import rt_to_T, invert_T, decompose_T, avg_quaternions
from tracker.tag_graph import TagGraph, dump_graph_as_extrinsics, seed_graph_from_extrinsics
from tracker.fusion import fuse_head_pose
from tracker.unity import UdpSender
from tracker.visualize import put_hud
from tracker.metrics import compute_metrics
metrics_on = True   # M ile aç/kapat


# ================== Kullanıcı ayarları (sende nasılsa öyle) ==================
SEND_CONFIDENCE_TO_UNITY = True
HOLOLENS_Y_COMP = 0.107414/2    # tvec Y kompanzasyonu (senin sabitin)

# ----------------- Kamera kalibrasyonu -----------------
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]

# ================== UDP ==================
udp = UdpSender(UDP_IP, UDP_PORT)

# ================== CSV ==================
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
csv_file = open(CSV_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time','TagIDs','FusedTx','FusedTy','FusedTz',
                     'R00','R01','R02','R10','R11','R12','R20','R21','R22',
                     'AvgDecisionMargin','Confidence','RScale'])

# ================== AprilTag dedektörü ==================
detector = PTagDetector(camera_matrix, dist_coeffs, FAMILY)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# ================== Filtre/Kalibrasyon durumları ==================
use_filter = True
calibrating = False
decision_margins_all = []

# Köprü kalibrasyon grafı (root = FRONT_TAG_ID=head)
graph = TagGraph(root_id=FRONT_TAG_ID, ema_alpha=EMA_ALPHA)

# Varsa eski extrinsics’i graf’a tohumla
loaded_extr = seed_graph_from_extrinsics(graph, EXTRINSICS_FILE)
pair_max  = None 
cycle_max = None
while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- 1) Detect (senin akışın: önce default, sonra ID boyuyla tekrar) ----
    raw_list = detector.detect_with_per_id_size(gray, TAG_SIZES)

    # ---- 2) detections dict’ini senin formatında kur + AXIS ÇİZ ----
    detections = {}
    for rd in raw_list:
        tid   = rd["id"]
        rmat  = rd["rmat"]
        tvec  = rd["tvec"].copy()
        dm    = rd["dm"]
        center= rd["center"]

        # SENİN KOMPANZASYONUN
        tvec_comp = tvec.copy()  # BURADA EKLEME YOK
        T_c_t = rt_to_T(rmat, tvec_comp)

        detections[tid] = {
            "T_c_t": T_c_t,
            "dm": dm,
            "center": center,
            "rmat": rmat,
            "tvec": tvec_comp
        }

        # === AXIS ÇİZİMİ SENDEKİ GİBİ ===
        rvec, _ = cv2.Rodrigues(rmat)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec_comp, 0.03)
        cv2.putText(frame, f'ID:{tid}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # ---- 3) Köprü kalibrasyonu (graf güncelleme) ----
    if calibrating:
        # aynı karede görünen her çift için kenar ekle/yumuşat
        graph.update_with_detections(detections)

    # ---- 4) Head pozu: reach edilebilen tag’lerden füzyon + Kalman ----
    fused = fuse_head_pose(detections, graph, use_filter=use_filter)
    
    if fused is not None:
        fused_pos, fused_quat, confidence, r_scale, used_ids, avg_dm = fused

        # Unity’ye gönder (senin işaret konv.)
        # ---- OpenCV(x,+y↓,z) --> Unity(x,+y↑,z) dönüşümü ----
        # 1) ÇIKIŞTA Y-ofset uygula
        p = fused_pos + np.array([0.0, HOLOLENS_Y_COMP, 0.0])
        # 2) Eksen flip matrisi (y'yi ters çevir)
        F = np.diag([1.0, -1.0, 1.0])
        # 3) Pozisyon dönüşümü
        unity_pos = (F @ p)
        # 4) Oryantasyon dönüşümü: R_u = F * R_cv * F
        R_cv = R.from_quat(fused_quat).as_matrix()
        R_u  = F @ R_cv @ F
        unity_quat = R.from_matrix(R_u).as_quat()

        udp.send({
            "timestamp": time.time(),
            "id": -1,
            "positionDif": unity_pos.tolist(),
            "rotationDif": unity_quat.tolist()
        })

        # CSV
        rmat_fused = R.from_quat(fused_quat).as_matrix()
        csv_writer.writerow([time.time(), used_ids] + list(fused_pos) +
                            list(rmat_fused.flatten()) +
                            [avg_dm, confidence, r_scale])

        # Doğrulama: front görünüyorsa fark
        if FRONT_TAG_ID in detections:
            Rf, tf = detections[FRONT_TAG_ID]["rmat"], detections[FRONT_TAG_ID]["tvec"]
            Rdiff = R.from_matrix(Rf).inv() * R.from_quat(fused_quat)
            ang_err = np.degrees(2*np.arccos(np.clip(Rdiff.as_quat()[-1], -1, 1)))
            pos_err = np.linalg.norm(tf - fused_pos)
        else:
            ang_err, pos_err = None, None
        # front görünüyorsa fark
        front_err_tuple = (ang_err, pos_err) if (ang_err is not None and pos_err is not None) else None

        metrics = None
        if metrics_on:
            metrics = compute_metrics(detections, graph, front_err=front_err_tuple)
        pair_max  = None if metrics is None else metrics["pair_max"]
        cycle_max = None if metrics is None else metrics["cycle_max"]

    else:
        used_ids, avg_dm, ang_err, pos_err = [], 0.0, None, None
        fused_pos = None

    # ---- 5) HUD ----
    put_hud(frame, fused_pos, used_ids, use_filter, calibrating, avg_dm,
        ang_err, pos_err, graph.reachable_nodes(),
        pair_max=pair_max, cycle_max=cycle_max)

    cv2.imshow("AprilTag Head Tracker", frame)

    # ---- 6) istatistik ----
    for d in detections.values():
        decision_margins_all.append(d["dm"])

    # ---- 7) Tuşlar ----
    key = cv2.waitKey(1)
    if key == 27:   # ESC
        break
    elif key in [ord('f'), ord('F')]:
        use_filter = not use_filter
    elif key in [ord('m'), ord('M')]:
        metrics_on = not metrics_on
    elif key in [ord('k'), ord('K')]:
        calibrating = not calibrating
    elif key in [ord('r'), ord('R')]:
        # extrinsics reset
        try:
            os.remove(EXTRINSICS_FILE)
            print("Extrinsics reset: file removed.")
        except Exception:
            pass
    elif key in [ord('s'), ord('S')]:
        # grafı tag->head extrinsics’e dök ve kaydet
        extr = dump_graph_as_extrinsics(graph)
        with open(EXTRINSICS_FILE, "w") as f:
            json.dump(extr, f, indent=2)
    elif key in [ord('l'), ord('L')]:
        seed_graph_from_extrinsics(graph, EXTRINSICS_FILE)
    elif key in [ord('v'), ord('V')]:
        # TagGraph'ın mevcut halini 3D olarak görselleştir
        try:
            graph.visualize_graph()
        except Exception as e:
            print(f"Görselleştirme hatası: {e}")

# ----------------- Cleanup -----------------
cap.release()
csv_file.close()
cv2.destroyAllWindows()

if len(decision_margins_all) > 0:
    print("=== Decision Margin Stats ===")
    print(f"Total: {len(decision_margins_all)}  "
          f"Mean: {np.mean(decision_margins_all):.2f}  "
          f"Std: {np.std(decision_margins_all):.2f}  "
          f"Min: {np.min(decision_margins_all):.2f}  "
          f"Max: {np.max(decision_margins_all):.2f}")

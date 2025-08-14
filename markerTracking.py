# -*- coding: utf-8 -*-
import cv2
import numpy as np
import mediapipe as mp
import csv
from datetime import datetime
import socket
import json
import time
from scipy.spatial.transform import Rotation as R
from kalmanFilter import PoseKalmanFilter

# =========================
#   ENGINE / EKSEN AYARLARI
# =========================
ENGINE = "unity"   # "unity" veya "kanzi"

def cv_to_engine_RT(R_cv: np.ndarray, t_cv: np.ndarray):
    S = np.diag([1.0, -1.0, 1.0])  # y eksenini tersle
    R_eng = S @ R_cv @ S
    t_eng = (S @ t_cv.reshape(3,1)).ravel()
    return R_eng, t_eng

def quat_for_engine(R_in: np.ndarray, engine: str):
    q_xyzw = R.from_matrix(R_in).as_quat()  # (x,y,z,w)
    if engine.lower() == "kanzi":
        x, y, z, w = q_xyzw
        return np.array([w, x, y, z])       # wxyz
    return q_xyzw                            # xyzw (Unity)

# =========================
#  FaceMesh + PnP AYARLARI
# =========================
# PnP için (burun, çene, kulak üstleri, ağız köşeleri) — mevcut akış
IDX_PNP = [1, 152, 162, 389, 61, 291]

# 3B model noktaları (METRE cinsinden yaklaşık kafa ölçüleri, sırası IDX_PNP ile aynı)
model_points = np.array([
    ( 0.000,  0.000,  0.000),   # 1   - Burun ucu
    ( 0.000, -0.090, -0.030),   # 152 - Çene
    (-0.065,  0.040, -0.050),   # 162 - Sol kulak üstü (yaklaşık/temporal)
    ( 0.065,  0.040, -0.050),   # 389 - Sağ kulak üstü (yaklaşık/temporal)
    (-0.040, -0.040, -0.030),   # 61  - Sol ağız köşesi
    ( 0.040, -0.040, -0.030)    # 291 - Sağ ağız köşesi
], dtype=np.float64)

# İSTENEN SALIENT NOKTALAR (çizim için)
FACE_IDS = [  # 15 salient facial points
    1, 33, 263, 61, 291, 199, 234, 454, 127,
    356, 152, 168, 94, 323, 93,
]

# AprilTag-benzeri kapsül
class FaceTag:
    def __init__(self, tag_id, Rmat, tvec, center, decision_margin):
        self.tag_id = tag_id
        self.pose_R = Rmat                  # 3x3
        self.pose_t = tvec.reshape(3, 1)    # 3x1
        self.center = center                # (u,v) piksel
        self.decision_margin = decision_margin

# =========================

# UDP hedef
UDP_IP = "127.0.0.1"
UDP_PORT = 12345
SEND_CONFIDENCE_TO_UNITY = True
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Kamera kalibrasyonu
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

tag_size = 0.08  # (akış korunması için)

# CSV
csv_file = open('apriltag_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'ID', 'Tx', 'Ty', 'Tz',
                     'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
                     'BetaX','BetaY','BetaZ','AlphaX','AlphaY','AlphaZ', 'DecisionMargin', 'Confidence', 'RScale'])

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Kalman
filters = {}
use_filter = True

# Confidence istatistik
decision_margins = []
show_confidence_details = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # akış uyumu

    # FaceMesh
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    # "tags" benzeri çıktı
    tags = []
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # --- SALIENT NOKTALARI ÇİZ ---
        for idx in FACE_IDS:
            if 0 <= idx < len(lm):
                u = int(lm[idx].x * w)
                v = int(lm[idx].y * h)
                cv2.circle(frame, (u, v), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(frame, str(idx), (u+3, v-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # PnP noktaları (mevcut akış)
        image_points = np.array([(lm[i].x * w, lm[i].y * h) for i in IDX_PNP], dtype=np.float64)

        ok, rvec_raw, tvec_raw = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                              flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            R_cv_raw, _ = cv2.Rodrigues(rvec_raw)

            # Reprojection error -> decision margin (yüksek iyi)
            proj, _ = cv2.projectPoints(model_points, rvec_raw, tvec_raw, camera_matrix, dist_coeffs)
            err = np.linalg.norm(image_points - proj.reshape(-1, 2), axis=1)
            rmse = float(np.sqrt(np.mean(err ** 2)))
            decision_margin = 1000.0 / (1.0 + rmse)

            center = tuple(map(int, image_points[0]))  # burun ucu

            tags = [FaceTag(tag_id=0, Rmat=R_cv_raw, tvec=tvec_raw.reshape(3), center=center,
                            decision_margin=decision_margin)]

    # --- AprilTag benzeri döngü ---
    for tag in tags:
        tag_id = tag.tag_id
        
        rmat_meas_cv = tag.pose_R
        tvec_meas_cv = tag.pose_t.reshape(3)
        quat_meas_cv = R.from_matrix(rmat_meas_cv).as_quat()

        decision_margin = tag.decision_margin
        decision_margins.append(decision_margin)
        
        if use_filter:
            if tag_id not in filters:
                filters[tag_id] = PoseKalmanFilter(tvec_meas_cv, quat_meas_cv)
            current_pos_cv, current_quat_cv, confidence, r_scale = filters[tag_id].update(
                tvec_meas_cv, quat_meas_cv, decision_margin
            )
        else:
            current_pos_cv = tvec_meas_cv
            current_quat_cv = quat_meas_cv
            confidence = None
            r_scale = None

        # HoloLens Y offset (CV)
        compensated_pos_cv = current_pos_cv.copy()
        compensated_pos_cv[1] += 0.107414/2

        # CV -> ENGINE
        R_cv_filt = R.from_quat(current_quat_cv).as_matrix()
        R_eng, pos_eng = cv_to_engine_RT(R_cv_filt, compensated_pos_cv)

        # Euler (ENGINE) — yaw/pitch işaret düzeltmesi
        sy = np.sqrt(R_eng[0,0]**2 + R_eng[1,0]**2)
        yaw   = -np.degrees(np.arctan2(R_eng[2,1], R_eng[2,2]))   # işaret tersine
        pitch = -np.degrees(np.arctan2(-R_eng[2,0], sy))          # işaret tersine
        roll  =  np.degrees(np.arctan2(R_eng[1,0], R_eng[0,0]))

        # UDP için quaternion (ENGINE uzayında, düzeltilmiş Euler'den)
        R_eng_fixed = R.from_euler('yxz', [pitch, yaw, roll], degrees=True).as_matrix()
        q_send = quat_for_engine(R_eng_fixed, ENGINE)
        pos_send = pos_eng

        # Görselleştirme ekseni (CV)
        rvec_draw, _ = cv2.Rodrigues(R_cv_filt)
        axis_length = 0.03
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_draw, compensated_pos_cv, axis_length)

        timestamp = time.time()

        # CSV (R: CV uzayı; açı: ENGINE Euler)
        row = [timestamp, tag_id] + list(current_pos_cv) + list(R_cv_filt.flatten()) + \
              [roll, pitch, yaw, roll, pitch, yaw, decision_margin, confidence, r_scale]
        csv_writer.writerow(row)

        # UDP gönder
        tag_data = {
            "timestamp": timestamp,
            "id": int(tag_id),
            "positionDif": pos_send.tolist(),
            "rotationDif": q_send.tolist()
        }
        sock.sendto(json.dumps(tag_data).encode(), (UDP_IP, UDP_PORT))

        # Overlay
        center = tuple(map(int, tag.center))
        cv2.putText(frame, f'ID: {tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Pos: {compensated_pos_cv[0]:.2f}, {compensated_pos_cv[1]:.2f}, {compensated_pos_cv[2]:.2f}', 
                    (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if show_confidence_details and confidence is not None:
            cv2.putText(frame, f'DM: {decision_margin:.1f} | Conf: {confidence:.2f}', 
                        (center[0], center[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f'R Scale: {r_scale:.2f}', 
                        (center[0], center[1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Statik overlayler
    filter_status = "ON" if use_filter else "OFF"
    cv2.putText(frame, f'Filter: {filter_status} (Press F to toggle)', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if len(decision_margins) > 0:
        recent_margins = decision_margins[-100:]
        avg_margin = np.mean(recent_margins)
        max_margin = np.max(recent_margins)
        min_margin = np.min(recent_margins)
        cv2.putText(frame, f'DM Stats - Avg: {avg_margin:.1f}, Max: {max_margin:.1f}, Min: {min_margin:.1f}', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'C: Toggle confidence details ({show_confidence_details})', 
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'T: Tune confidence params', 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Pencere adı aynı
    cv2.imshow("AprilTag Tracker", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('f') or key == ord('F'):
        use_filter = not use_filter
        print(f"Filter toggled: {'ON' if use_filter else 'OFF'}")
    elif key == ord('c') or key == ord('C'):
        show_confidence_details = not show_confidence_details
        print(f"Confidence details: {'ON' if show_confidence_details else 'OFF'}")
    elif key == ord('t') or key == ord('T'):
        if len(decision_margins) > 50:
            recent_margins = decision_margins[-200:]
            suggested_max = np.percentile(recent_margins, 95)
            print(f"\n=== Confidence Parameter Tuning ===")
            print(f"Recent Decision Margin Stats:")
            print(f"  Mean: {np.mean(recent_margins):.2f}")
            print(f"  Std: {np.std(recent_margins):.2f}")
            print(f"  95th percentile: {suggested_max:.2f}")
            print(f"  Current max_decision_margin: {filters[list(filters.keys())[0]].max_decision_margin if filters else 'N/A'}")
            print(f"Suggested max_decision_margin: {suggested_max}")
            for filter_obj in filters.values():
                filter_obj.set_confidence_params(max_decision_margin=suggested_max)
            print("Parameters updated automatically!")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
face_mesh.close()

# Final istatistikler
if len(decision_margins) > 0:
    print(f"\n=== Final Decision Margin Statistics ===")
    print(f"Total measurements: {len(decision_margins)}")
    print(f"Mean: {np.mean(decision_margins):.2f}")
    print(f"Std: {np.std(decision_margins):.2f}")
    print(f"Min: {np.min(decision_margins):.2f}")
    print(f"Max: {np.max(decision_margins):.2f}")
    print(f"95th percentile: {np.percentile(decision_margins, 95):.2f}")
    print(f"Recommended max_decision_margin: {np.percentile(decision_margins, 95):.2f}")

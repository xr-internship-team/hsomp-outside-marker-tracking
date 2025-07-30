import cv2
import numpy as np
from pupil_apriltags import Detector
import csv
from datetime import datetime
import socket
import json
import time
from scipy.spatial.transform import Rotation as R
from kalmanFilter import PoseKalmanFilter

# UDP hedef bilgileri
UDP_IP = "192.168.137.47"  # Unity çalışıyorsa localhost, değilse Unity IP adresi
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Kalibrasyon parametrelerini yükle
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

tag_size = 0.0375 # Metre cinsinden

csv_file = open('apriltag_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'ID', 'Tx', 'Ty', 'Tz',
                     'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
                     'BetaX','BetaY','BetaZ','AlphaX','AlphaY','AlphaZ'])

detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Her tag ID için ayrı Kalman filtresi sakla
filters = {}
use_filter = True  # Filtreleme aktif/pasif kontrolü için anahtar

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=[camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]],
                           tag_size=tag_size)

    for tag in tags:
        tag_id = tag.tag_id
        
        # Ham pozisyon ve rotasyon
        rmat = tag.pose_R
        tvec = tag.pose_t.reshape(3)
        r = R.from_matrix(rmat)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Filtreleme aktifse Kalman filtresini kullan
        if use_filter:
            # Bu tag için filtre yoksa oluştur
            if tag_id not in filters:
                filters[tag_id] = PoseKalmanFilter(tvec, quat)
            
            # Filtreyi güncelle
            current_pos, current_quat = filters[tag_id].update(tvec, quat)
        else:
            # Filtreleme pasifse ham verileri kullan
            current_pos = tvec
            current_quat = quat
        
        # Eksenleri çiz
        rvec, _ = cv2.Rodrigues(rmat)
        axis_length = 0.03
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, current_pos, axis_length)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # CSV'ye yaz (filtrelenmemiş ham veri)
        row = [timestamp, tag.tag_id] + list(tvec) + list(rmat.flatten())
        csv_writer.writerow(row)

        # Unity için dönüşüm
        unity_quat = current_quat * [1, -1, 1, 1]
        unity_pos = current_pos * [1, -1, 1]
        
        tag_data = {
            "timestamp": timestamp,
            "id": int(tag_id),
            "positionDif": unity_pos.tolist(),
            "rotationDif": unity_quat.tolist(),
        }

        message = json.dumps(tag_data)
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        # Görselleştirme
        center = tuple(map(int, tag.center))
        cv2.putText(frame, f'ID: {tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Pos: {current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}', 
                    (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Filtre durumunu ekranda göster
    filter_status = "ON" if use_filter else "OFF"
    cv2.putText(frame, f'Filter: {filter_status} (Press F to toggle)', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("AprilTag Tracker", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('f') or key == ord('F'):  # F tuşu ile filtreyi aç/kapa
        use_filter = not use_filter
        print(f"Filter toggled: {'ON' if use_filter else 'OFF'}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
import cv2
import numpy as np
from pupil_apriltags import Detector
import csv
from datetime import datetime
import socket
import json
import time
from scipy.spatial.transform import Rotation as R

# UDP hedef bilgileri
UDP_IP = "127.0.0.1"  # Unity çalışıyorsa localhost, değilse Unity IP adresi
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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=[camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]],
                           tag_size=tag_size)

    for tag in tags:

        rmat = tag.pose_R
        tvec = tag.pose_t.reshape(3)

        r = R.from_matrix(rmat)
        quat = r.as_quat()  # [x, y, z, w]

        # Dönüş matrisini rotasyon vektörüne çevir
        rvec, _ = cv2.Rodrigues(rmat)

        # AprilTag koordinat eksenlerini çiz (X: kırmızı, Y: yeşil, Z: mavi)
        axis_length = 0.03  # metre cinsinden eksen uzunluğu
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

        P_local = np.array([0.0, 1, 1])  # Marker göreceli nokta
        P_global = rmat @ P_local + tvec

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        row = [timestamp, tag.tag_id] + list(tvec) + list(rmat.flatten()) + list(P_global) + list(P_local)
        csv_writer.writerow(row)

        # Rotation matrix'i düzleştir
        rotation_matrix_flat = rmat.flatten().tolist()
        print(quat)
        quat*= [-1,1,-1,1]
        tvec *= [1, -1, 1]  # Unity için uygun hale getirme
        tag_data = {
            "timestamp": timestamp,
            "id": int(tag.tag_id),
            "translation": tvec.tolist(),
            "quaternion": quat.tolist(),
            "rotation_matrix_flat": rotation_matrix_flat,
            "beta_point": P_global.tolist(),
            "alpha_point": P_local.tolist()
        }

        message = json.dumps(tag_data)
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
        #print(f"UDP gönderildi: ID={tag.tag_id}, Zaman={timestamp}") 

        center = tuple(map(int, tag.center))
        cv2.putText(frame, f'ID: {tag.tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("AprilTag Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
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
UDP_IP = "10.10.50.63"  # Unity çalışıyorsa localhost, değilse Unity IP adresi
UDP_PORT = 12345

# Configuration for Unity data transmission
SEND_CONFIDENCE_TO_UNITY = True  # Set to False if Unity doesn't need confidence data

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Kalibrasyon parametrelerini yükle
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

tag_size = 0.08  # Metre cinsinden

csv_file = open('apriltag_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'ID', 'Tx', 'Ty', 'Tz',
                     'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
                     'BetaX','BetaY','BetaZ','AlphaX','AlphaY','AlphaZ', 'DecisionMargin', 'Confidence', 'RScale'])

detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Her tag ID için ayrı Kalman filtresi sakla
filters = {}
use_filter = True  # Filtreleme aktif/pasif kontrolü için anahtar

# Confidence tracking for tuning
decision_margins = []  # For statistics
show_confidence_details = False  # Toggle for detailed confidence display

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
        
        # Decision margin confidence değeri
        decision_margin = tag.decision_margin
        decision_margins.append(decision_margin)  # For statistics
        
        # Filtreleme aktifse Kalman filtresini kullan
        if use_filter:
            # Bu tag için filtre yoksa oluştur
            if tag_id not in filters:
                filters[tag_id] = PoseKalmanFilter(tvec, quat)
            
            # Filtreyi güncelle (decision_margin ile dinamik R ayarı)
            current_pos, current_quat, confidence, r_scale = filters[tag_id].update(tvec, quat, decision_margin)
        else:
            # Filtreleme pasifse ham verileri kullan
            current_pos = tvec
            current_quat = quat
            confidence = None
            r_scale = None
        
        # Eksenleri çiz
        rvec, _ = cv2.Rodrigues(rmat)
        
        # AprilTag koordinat eksenlerini çiz (X: kırmızı, Y: yeşil, Z: mavi)
        axis_length = 0.03
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, current_pos, axis_length)

        timestamp = time.time()

        # CSV'ye yaz (hem ham hem işlenmiş veri + confidence bilgileri)
        row = [timestamp, tag.tag_id] + list(tvec) + list(rmat.flatten()) + [decision_margin, confidence, r_scale]
        csv_writer.writerow(row)

        # HoloLens Y offset compensation (HoloLens is 0.107414m above tripod marker)
        # Apply in camera coordinate system where measurement was made
        compensated_pos = current_pos.copy()
        compensated_pos[1] += 0.107414/2  # Add offset for HoloLens height above tripod
        
        # Unity için dönüşüm (X ve Z eksenleri düzeltildi)
        unity_quat = current_quat * [-1, -1, -1, 1]  # X, Y, Z bileşenleri çevrildi
        unity_pos = compensated_pos * [1, 1, 1]        # X, Y, Z eksenleri çevrildi
        
        tag_data = {
            "timestamp": timestamp,
            "id": int(tag_id),
            "positionDif": unity_pos.tolist(),
            "rotationDif": unity_quat.tolist()
        }
        
        # Optionally add confidence data for Unity
        if SEND_CONFIDENCE_TO_UNITY:
            tag_data["confidence"] = float(confidence) if confidence is not None else None
            tag_data["decision_margin"] = float(decision_margin)

        message = json.dumps(tag_data)
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
        #print(f"UDP gönderildi: ID={tag.tag_id}, Zaman={timestamp}") 

        # Görselleştirme
        center = tuple(map(int, tag.center))
        cv2.putText(frame, f'ID: {tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Pos: {compensated_pos[0]:.2f}, {compensated_pos[1]:.2f}, {compensated_pos[2]:.2f}', 
                    (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Confidence ve decision margin bilgilerini göster
        if show_confidence_details and confidence is not None:
            cv2.putText(frame, f'DM: {decision_margin:.1f} | Conf: {confidence:.2f}', 
                        (center[0], center[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f'R Scale: {r_scale:.2f}', 
                        (center[0], center[1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Filtre durumunu ve confidence istatistiklerini ekranda göster
    filter_status = "ON" if use_filter else "OFF"
    cv2.putText(frame, f'Filter: {filter_status} (Press F to toggle)', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Decision margin istatistikleri (son 100 ölçüm)
    if len(decision_margins) > 0:
        recent_margins = decision_margins[-100:]  # Son 100 ölçüm
        avg_margin = np.mean(recent_margins)
        max_margin = np.max(recent_margins)
        min_margin = np.min(recent_margins)
        
        cv2.putText(frame, f'DM Stats - Avg: {avg_margin:.1f}, Max: {max_margin:.1f}, Min: {min_margin:.1f}', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Kontrol tuşları açıklaması
    cv2.putText(frame, f'C: Toggle confidence details ({show_confidence_details})', 
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'T: Tune confidence params', 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("AprilTag Tracker", frame)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("AprilTag Tracker", cv2.WND_PROP_VISIBLE) < 1:  # ESC
        break
    elif key == ord('f') or key == ord('F'):  # F tuşu ile filtreyi aç/kapa
        use_filter = not use_filter
        print(f"Filter toggled: {'ON' if use_filter else 'OFF'}")
    elif key == ord('c') or key == ord('C'):  # C tuşu ile confidence detaylarını aç/kapa
        show_confidence_details = not show_confidence_details
        print(f"Confidence details: {'ON' if show_confidence_details else 'OFF'}")
    elif key == ord('t') or key == ord('T'):  # T tuşu ile confidence parametrelerini ayarla
        if len(decision_margins) > 50:  # Yeterli veri varsa
            recent_margins = decision_margins[-200:]
            suggested_max = np.percentile(recent_margins, 95)  # %95'lik percentile
            print(f"\n=== Confidence Parameter Tuning ===")
            print(f"Recent Decision Margin Stats:")
            print(f"  Mean: {np.mean(recent_margins):.2f}")
            print(f"  Std: {np.std(recent_margins):.2f}")
            print(f"  95th percentile: {suggested_max:.2f}")
            print(f"  Current max_decision_margin: {filters[list(filters.keys())[0]].max_decision_margin if filters else 'N/A'}")
            print(f"Suggested max_decision_margin: {suggested_max}")
            
            # Otomatik ayarlama
            for filter_obj in filters.values():
                filter_obj.set_confidence_params(max_decision_margin=suggested_max)
            print("Parameters updated automatically!")

cap.release()
csv_file.close()
cv2.destroyAllWindows()

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
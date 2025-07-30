import cv2
import numpy as np
from pupil_apriltags import Detector
import csv
from datetime import datetime
import socket
import json
import time
from scipy.spatial.transform import Rotation as R
from kalman_filter import AprilTagKalmanFilter

# UDP hedef bilgileri
UDP_IP = "127.0.0.1"  # Unity çalışıyorsa localhost, değilse Unity IP adresi
UDP_PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Kalibrasyon parametrelerini yükle
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

tag_size = 0.0375  # Metre cinsinden

# CSV dosyasını aç (raw ve processed verileri kaydeder)
csv_file = open('apriltag_log_kalman.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'ID', 'Raw_Tx', 'Raw_Ty', 'Raw_Tz', 'Raw_Qx', 'Raw_Qy', 'Raw_Qz', 'Raw_Qw',
                     'Processed_Tx', 'Processed_Ty', 'Processed_Tz', 'Processed_Qx', 'Processed_Qy', 'Processed_Qz', 'Processed_Qw',
                     'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22', 'Filtering_Enabled'])

detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

# Kalman filter'ı başlat
# use_position=True: hem pozisyon hem orientasyon filtreler (7D state)
# use_position=False: sadece orientasyon filtreler (4D state)
kalman_tracker = AprilTagKalmanFilter(use_position=True, dt=1.0/30.0)

# Filtering control
filtering_enabled = True  # Flag to enable/disable filtering
print("Kalman Filter Controls:")
print("  'f' key: Toggle filtering ON/OFF")
print("  'r' key: Reset all filters")
print("  ESC key: Exit program")
print("-" * 40)

# Görülen tag'leri takip etmek için
detected_tags_current = set()
detected_tags_previous = set()
lost_tag_threshold = 10  # Kaç frame kayıp olursa filter'ı sil

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_count = 0
missing_tag_counts = {}  # Her tag için kaç frame boyunca kayıp olduğunu takip eder

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # AprilTag detection
    tags = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=[camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]],
                           tag_size=tag_size)

    detected_tags_current.clear()
    
    for tag in tags:
        detected_tags_current.add(tag.tag_id)
        
        # Reset missing count for detected tag
        if tag.tag_id in missing_tag_counts:
            del missing_tag_counts[tag.tag_id]

        # Raw pose extraction
        rmat = tag.pose_R
        tvec = tag.pose_t.reshape(3)

        r = R.from_matrix(rmat)
        quat_raw = r.as_quat()  # [x, y, z, w] format from scipy

        # Unity koordinat sistemine çevir
        quat_raw *= [-1, 1, -1, 1]
        tvec_raw = tvec * [1, -1, 1]  # Unity için uygun hale getirme

        # Decide whether to use filtering or raw data
        if filtering_enabled:
            # Kalman filter için measurement hazırla
            # scipy quaternion format: [x, y, z, w] -> [w, x, y, z] format'a çevir
            quat_wxyz = np.array([quat_raw[3], quat_raw[0], quat_raw[1], quat_raw[2]])
            
            # Measurement vector: [x, y, z, q_w, q_x, q_y, q_z]
            measurement = np.concatenate([tvec_raw, quat_wxyz])
            
            # Kalman filter ile güncelle
            filtered_state = kalman_tracker.update(tag.tag_id, measurement)
            
            # Filtrelenmiş değerleri çıkar
            filtered_tvec = filtered_state[:3]
            filtered_quat_wxyz = filtered_state[3:7]
            
            # Unity için quaternion format'a geri çevir [w,x,y,z] -> [x,y,z,w]
            filtered_quat = np.array([filtered_quat_wxyz[1], filtered_quat_wxyz[2], 
                                     filtered_quat_wxyz[3], filtered_quat_wxyz[0]])
        else:
            # Use raw measurements directly (no filtering)
            filtered_tvec = tvec_raw.copy()
            filtered_quat = quat_raw.copy()

        # Rotation vektörü hesapla (çizim için)
        filtered_rmat = R.from_quat(filtered_quat).as_matrix()
        rvec, _ = cv2.Rodrigues(filtered_rmat)

        # AprilTag koordinat eksenlerini çiz (filtrelenmiş pose ile)
        axis_length = 0.03  # metre cinsinden eksen uzunluğu
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, filtered_tvec, axis_length)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # CSV'ye hem raw hem processed verileri kaydet
        row = [timestamp, tag.tag_id] + list(tvec_raw) + list(quat_raw) + list(filtered_tvec) + list(filtered_quat) + list(rmat.flatten()) + [filtering_enabled]
        csv_writer.writerow(row)

        # UDP ile veriyi gönder (filtered veya raw)
        tag_data = {
            "timestamp": timestamp,
            "id": int(tag.tag_id),
            "translation": filtered_tvec.tolist(),
            "quaternion": filtered_quat.tolist(),
            "filtered": filtering_enabled  # Indicates if data is filtered or raw
        }

        message = json.dumps(tag_data)
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        # Tag bilgisini ekranda göster
        center = tuple(map(int, tag.center))
        status_text = f'ID: {tag.tag_id} ({"Filtered" if filtering_enabled else "Raw"})'
        color = (0, 255, 0) if filtering_enabled else (0, 165, 255)  # Green for filtered, Orange for raw
        cv2.putText(frame, status_text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Kayıp tag'ler için prediction-only step (sadece filtering aktifken)
    if filtering_enabled:
        missing_tags = detected_tags_previous - detected_tags_current
        
        for tag_id in missing_tags:
            # Kayıp sayısını artır
            if tag_id not in missing_tag_counts:
                missing_tag_counts[tag_id] = 0
            missing_tag_counts[tag_id] += 1
            
            # Eğer çok uzun süredir kayıpsa filter'ı sil
            if missing_tag_counts[tag_id] > lost_tag_threshold:
                kalman_tracker.remove_filter(tag_id)
                del missing_tag_counts[tag_id]
                print(f"Tag {tag_id} filter removed after {lost_tag_threshold} frames")
            else:
                # Sadece prediction yap
                predicted_state = kalman_tracker.predict_only(tag_id)
                if predicted_state is not None:
                    # Tahmin edilen pose'u göster (isteğe bağlı)
                    predicted_tvec = predicted_state[:3]
                    predicted_quat_wxyz = predicted_state[3:7]
                    
                    # Unity format'a çevir
                    predicted_quat = np.array([predicted_quat_wxyz[1], predicted_quat_wxyz[2], 
                                              predicted_quat_wxyz[3], predicted_quat_wxyz[0]])
                    
                    # Tahmin edilen pose'u UDP ile gönder (isteğe bağlı)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    tag_data = {
                        "timestamp": timestamp,
                        "id": int(tag_id),
                        "translation": predicted_tvec.tolist(),
                        "quaternion": predicted_quat.tolist(),
                        "predicted": True  # Bu tahmin edilmiş veri olduğunu belirt
                    }
                    
                    message = json.dumps(tag_data)
                    sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
                    
                    print(f"Tag {tag_id} predicted (missing for {missing_tag_counts[tag_id]} frames)")
    else:
        # Clear missing tag counts when filtering is disabled
        missing_tag_counts.clear()

    # Önceki frame'deki tag listesini güncelle
    detected_tags_previous = detected_tags_current.copy()

    # Display status information
    active_filters = len(kalman_tracker.filters)
    status_color = (0, 255, 0) if filtering_enabled else (0, 165, 255)
    
    cv2.putText(frame, f'Filtering: {"ON" if filtering_enabled else "OFF"}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f'Active Filters: {active_filters}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'F: Toggle Filter | R: Reset | ESC: Exit', (10, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("AprilTag Tracker with Kalman Filter", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC tuşu
        break
    elif key == ord('f') or key == ord('F'):  # Toggle filtering
        filtering_enabled = not filtering_enabled
        status = "ENABLED" if filtering_enabled else "DISABLED"
        print(f"Kalman filtering {status}")
    elif key == ord('r') or key == ord('R'):  # Reset all filters
        # Remove all existing filters
        kalman_tracker.filters.clear()
        missing_tag_counts.clear()
        print("All Kalman filters RESET")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("Tracking completed. Data saved to 'apriltag_log_kalman.csv'")
print("CSV includes both raw and processed data with filtering status.") 
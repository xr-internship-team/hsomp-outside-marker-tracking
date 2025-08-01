import cv2
import numpy as np
from pupil_apriltags import Detector
import csv
from datetime import datetime
import socket
import json
import time
import os
from scipy.spatial.transform import Rotation as R
from kalmanFilter import PoseKalmanFilter   
import tkinter as tk
from tkinter import simpledialog

# Kalibrasyon parametrelerini yükle
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]


# Ana pencereyi gizle (yalnızca popup görünsün)
root = tk.Tk()
root.withdraw()

# Kullanıcıdan input al
tag_size = simpledialog.askstring(title="Input", prompt="Tag size:")

print("Kullanıcı girdi:", tag_size)
tag_size = float(tag_size)

# Data collection control
collecting_data = False
data_to_save = []
target_count = 100
collection_number = 0

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
        tag_id = tag.tag_id
        
        # Ham pozisyon ve rotasyon
        rmat = tag.pose_R
        tvec = tag.pose_t.reshape(3)
        r = R.from_matrix(rmat)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Eksenleri çiz
        rvec, _ = cv2.Rodrigues(rmat)
        
        # AprilTag koordinat eksenlerini çiz (X: kırmızı, Y: yeşil, Z: mavi)
        axis_length = 0.03
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Collect data if collection is active
        if collecting_data:
            row = [timestamp, collection_number, tag.tag_id] + list(tvec) + list(rmat.flatten()) + [tag_size]
            data_to_save.append(row)
            print(f"Collecting data: {len(data_to_save)}/{target_count} (Collection #{collection_number})")
            
            # Check if we've collected enough data
            if len(data_to_save) >= target_count:
                print(f"Collected {target_count} data points for collection #{collection_number}, saving to CSV...")
                
                # Check if CSV file exists to determine if we need to write header
                file_exists = os.path.exists('apriltag_log.csv')
                
                # Save to CSV (append mode)
                with open('apriltag_log.csv', mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    
                    # Write header only if file doesn't exist
                    if not file_exists:
                        csv_writer.writerow(['Time', 'Collection_Number', 'ID', 'Tx', 'Ty', 'Tz',
                                            'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
                                            'tag_size'])
                    
                    csv_writer.writerows(data_to_save)
                
                print(f"Data saved successfully! Collection #{collection_number} completed.")
                
                # Reset collection state
                collecting_data = False
                data_to_save = []

        # Unity için dönüşüm (X ve Z eksenleri düzeltildi)
        #unity_quat = quat * [-1, -1, -1, 1]  # X, Y, Z bileşenleri çevrildi
        #unity_pos = tvec * [1, 1, 1]        # X, Y, Z eksenleri çevrildi
        
        """ tag_data = {
            "timestamp": timestamp,
            "id": int(tag_id),
            "positionDif": unity_pos.tolist(),
            "rotationDif": unity_quat.tolist()
        } """

        # Görselleştirme
        center = tuple(map(int, tag.center))
        cv2.putText(frame, f'ID: {tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Pos: {tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}', 
                    (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Show data collection status
    if collecting_data:
        status_text = f"COLLECTING #{collection_number}: {len(data_to_save)}/{target_count}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        next_collection = collection_number + 1
        cv2.putText(frame, f"Press 'S' to start collection #{next_collection}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("AprilTag Tracker", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and not collecting_data:
        collection_number += 1
        print(f"Starting data collection #{collection_number} - will collect next 100 data points...")
        collecting_data = True
        data_to_save = []
    elif key == 27:  # ESC
        break
    
cap.release()
cv2.destroyAllWindows()

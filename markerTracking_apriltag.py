import cv2
import numpy as np
from pupil_apriltags import Detector
import csv
import socket
import json
import time
from scipy.spatial.transform import Rotation as R
from kalmanFilter import PoseKalmanFilter

# UDP target information
UDP_IP = "127.0.0.1"  # Localhost for testing, change to HoloLens IP for deployment
UDP_PORT = 12345

# Enable or disable CSV logging
ENABLE_CSV_LOG = True

CAMERA_INDEX = 0  # Default camera index, change if needed

# AprilTag parameters
tag_size = 0.08  # Tag size in meters

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load camera calibration parameters
with np.load("calib_params.npz") as data:
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

if ENABLE_CSV_LOG:
    csv_file = open('apriltag_log.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time', 'ID', 'Tx', 'Ty', 'Tz',
                         'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22',
                         'BetaX','BetaY','BetaZ','AlphaX','AlphaY','AlphaZ', 'DecisionMargin', 'Confidence', 'RScale'])

# Initialize AprilTag detector
detector = Detector(families="tag36h11",
                    nthreads=1,
                    quad_decimate=1,
                    quad_sigma=0.0,
                    refine_edges=1,
                    decode_sharpening=0.25,
                    debug=0)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Store a separate Kalman filter for each tag ID
filters = {}
use_filter = True  # Toggle filtering on/off

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
        
        # Raw position and rotation
        rmat = tag.pose_R
        tvec = tag.pose_t.reshape(3)
        r = R.from_matrix(rmat)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Decision margin (confidence value)
        decision_margin = tag.decision_margin
        decision_margins.append(decision_margin)  # For statistics
        
        # Use Kalman filter if enabled
        if use_filter:
            # Create filter if it doesn't exist for this tag
            if tag_id not in filters:
                filters[tag_id] = PoseKalmanFilter(tvec, quat)
            
            # Update filter (dynamic R adjustment with decision_margin)
            current_pos, current_quat, confidence, r_scale = filters[tag_id].update(tvec, quat, decision_margin)
        else:
            # Use raw data if filtering is disabled
            current_pos = tvec
            current_quat = quat
            confidence = None
            r_scale = None

        
        # Draw axes
        rvec, _ = cv2.Rodrigues(rmat)
        
        # Draw AprilTag coordinate axes (X: red, Y: green, Z: blue)
        axis_length = 0.03
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, current_pos, axis_length)

        timestamp = time.time()
        # Write to CSV (both raw and filtered data + confidence info)
        # Euler açılarını hesapla (derece cinsinden)
        euler_deg = r.as_euler('xyz', degrees=True)
        betaX, betaY, betaZ = euler_deg
        alphaX, alphaY, alphaZ = euler_deg  # Aynı sistem kullanılıyorsa tekrar kullanılabilir

        row = [timestamp, tag.tag_id] + list(tvec) + list(rmat.flatten()) + \
              [betaX, betaY, betaZ, alphaX, alphaY, alphaZ, decision_margin, confidence, r_scale]
  
       # row = [timestamp, tag.tag_id] + list(tvec) + list(rmat.flatten()) + [decision_margin, confidence, r_scale]
        if ENABLE_CSV_LOG:
            csv_writer.writerow(row)

        # Unity conversion (invert X, Y, Z axes as needed)
        unity_quat = current_quat * [-1, -1, -1, 1]  # Invert X, Y, Z components
        unity_pos = current_pos * [1, 1, 1]      # Invert axes if required
        
        tag_data = {
            "timestamp": timestamp,
            "id": int(tag_id),
            "positionDif": unity_pos.tolist(),
            "rotationDif": unity_quat.tolist(),
        }

        message = json.dumps(tag_data)
        sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

        # Visualization
        center = tuple(map(int, tag.center))
        cv2.putText(frame, f'ID: {tag_id}', center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'Pos: {current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}', 
                    (center[0], center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show confidence and decision margin info
        if show_confidence_details and confidence is not None:
            cv2.putText(frame, f'DM: {decision_margin:.1f} | Conf: {confidence:.2f}', 
                        (center[0], center[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f'R Scale: {r_scale:.2f}', 
                        (center[0], center[1] + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Show filter status and confidence statistics on screen
    filter_status = "ON" if use_filter else "OFF"
    cv2.putText(frame, f'Filter: {filter_status} (Press F to toggle)', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Decision margin statistics (last 100 measurements)
    if len(decision_margins) > 0:
        recent_margins = decision_margins[-100:]  # Last 100 measurements
        avg_margin = np.mean(recent_margins)
        max_margin = np.max(recent_margins)
        min_margin = np.min(recent_margins)
        
        cv2.putText(frame, f'DM Stats - Avg: {avg_margin:.1f}, Max: {max_margin:.1f}, Min: {min_margin:.1f}', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Control key explanations
    cv2.putText(frame, f'C: Toggle confidence details ({show_confidence_details})', 
                (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f'T: Tune confidence params', 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("AprilTag Tracker", frame)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("AprilTag Tracker", cv2.WND_PROP_VISIBLE) < 1:  # ESC
        break
    elif key == ord('f') or key == ord('F'):  # Toggle filter with F key
        use_filter = not use_filter
        print(f"Filter toggled: {'ON' if use_filter else 'OFF'}")
    elif key == ord('c') or key == ord('C'):  # Toggle confidence details with C key
        show_confidence_details = not show_confidence_details
        print(f"Confidence details: {'ON' if show_confidence_details else 'OFF'}")
    elif key == ord('t') or key == ord('T'):  # Tune confidence parameters with T key
        if len(decision_margins) > 50:  # If enough data
            recent_margins = decision_margins[-200:]
            suggested_max = np.percentile(recent_margins, 95)  # 95th percentile
            print(f"\n=== Confidence Parameter Tuning ===")
            print(f"Recent Decision Margin Stats:")
            print(f"  Mean: {np.mean(recent_margins):.2f}")
            print(f"  Std: {np.std(recent_margins):.2f}")
            print(f"  95th percentile: {suggested_max:.2f}")
            print(f"  Current max_decision_margin: {filters[list(filters.keys())[0]].max_decision_margin if filters else 'N/A'}")
            print(f"Suggested max_decision_margin: {suggested_max}")
            
            # Auto update parameters
            for filter_obj in filters.values():
                filter_obj.set_confidence_params(max_decision_margin=suggested_max)
            print("Parameters updated automatically!")

cap.release()
if ENABLE_CSV_LOG:
    csv_file.close()
cv2.destroyAllWindows()

# Final statistics
if len(decision_margins) > 0:
    print(f"\n=== Final Decision Margin Statistics ===")
    print(f"Total measurements: {len(decision_margins)}")
    print(f"Mean: {np.mean(decision_margins):.2f}")
    print(f"Std: {np.std(decision_margins):.2f}")
    print(f"Min: {np.min(decision_margins):.2f}")
    print(f"Max: {np.max(decision_margins):.2f}")
    print(f"95th percentile: {np.percentile(decision_margins, 95):.2f}")
    print(f"Recommended max_decision_margin: {np.percentile(decision_margins, 95):.2f}")
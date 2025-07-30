#!/usr/bin/env python
"""
Demo script showing the toggle functionality of AprilTag Kalman Filter
Uses simulated data and OpenCV window to demonstrate real-time control
"""

import cv2
import numpy as np
import time
from kalman_filter import AprilTagKalmanFilter
from scipy.spatial.transform import Rotation as R

def generate_noisy_measurement(t, noise_level=0.02):
    """Generate a simulated noisy AprilTag measurement"""
    # Simulate circular motion
    radius = 0.5
    true_pos = np.array([
        radius * np.cos(t),
        radius * np.sin(t),
        2.0 + 0.1 * np.sin(2*t)
    ])
    
    # Add position noise
    noisy_pos = true_pos + np.random.normal(0, noise_level, 3)
    
    # Simulate rotation around Z axis
    true_rot = R.from_euler('z', t)
    # Add small rotation noise
    noise_rot = R.from_euler('xyz', np.random.normal(0, noise_level, 3))
    noisy_rot = true_rot * noise_rot
    
    # Convert to quaternion [w, x, y, z] format
    quat_xyzw = noisy_rot.as_quat()  # [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    
    return true_pos, noisy_pos, quat_wxyz

def draw_pose_info(frame, pos, quat, label, color, y_offset=0):
    """Draw pose information on the frame"""
    base_y = 50 + y_offset
    
    cv2.putText(frame, f'{label}:', (20, base_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f'Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})', 
               (20, base_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(frame, f'Quat: ({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})', 
               (20, base_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def main():
    print("AprilTag Kalman Filter Demo")
    print("=" * 40)
    print("Controls:")
    print("  'F' key: Toggle filtering ON/OFF")
    print("  'R' key: Reset filter")
    print("  'ESC' key: Exit")
    print("  'SPACE' key: Add noise burst")
    print("-" * 40)
    
    # Initialize filter
    kalman_tracker = AprilTagKalmanFilter(use_position=True, dt=1.0/30.0)
    
    # Demo parameters
    filtering_enabled = True
    tag_id = 0
    t = 0
    dt = 1.0/30.0
    noise_burst = False
    
    # Create window
    frame_width, frame_height = 800, 600
    
    while True:
        # Create frame
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Generate measurement
        noise_level = 0.05 if noise_burst else 0.02
        true_pos, noisy_pos, noisy_quat = generate_noisy_measurement(t, noise_level)
        
        # Reset noise burst after one frame
        if noise_burst:
            noise_burst = False
        
        # Create measurement vector
        measurement = np.concatenate([noisy_pos, noisy_quat])
        
        # Process measurement
        if filtering_enabled:
            # Use Kalman filter
            filtered_state = kalman_tracker.update(tag_id, measurement)
            processed_pos = filtered_state[:3]
            processed_quat = filtered_state[3:7]
            mode_text = "KALMAN FILTERED"
            mode_color = (0, 255, 0)  # Green
        else:
            # Use raw measurements
            processed_pos = noisy_pos
            processed_quat = noisy_quat
            mode_text = "RAW MEASUREMENTS"
            mode_color = (0, 165, 255)  # Orange
        
        # Draw title and mode
        cv2.putText(frame, "AprilTag Kalman Filter Demo", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode_text}", (20, frame_height - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Draw controls
        cv2.putText(frame, "F: Toggle | R: Reset | SPACE: Noise | ESC: Exit", 
                   (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw pose information
        draw_pose_info(frame, true_pos, [0, 0, 0, 1], "True Pose", (255, 255, 255), 0)
        draw_pose_info(frame, noisy_pos, noisy_quat, "Noisy Measurement", (0, 0, 255), 80)
        draw_pose_info(frame, processed_pos, processed_quat, "Processed Output", mode_color, 160)
        
        # Draw position trajectory (simple visualization)
        center_x, center_y = frame_width // 2, frame_height // 2 + 50
        scale = 200
        
        # True position
        true_px = int(center_x + true_pos[0] * scale)
        true_py = int(center_y + true_pos[1] * scale)
        cv2.circle(frame, (true_px, true_py), 5, (255, 255, 255), -1)
        cv2.putText(frame, "True", (true_px + 10, true_py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Noisy position
        noisy_px = int(center_x + noisy_pos[0] * scale)
        noisy_py = int(center_y + noisy_pos[1] * scale)
        cv2.circle(frame, (noisy_px, noisy_py), 3, (0, 0, 255), -1)
        cv2.putText(frame, "Noisy", (noisy_px + 10, noisy_py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Processed position
        proc_px = int(center_x + processed_pos[0] * scale)
        proc_py = int(center_y + processed_pos[1] * scale)
        cv2.circle(frame, (proc_px, proc_py), 4, mode_color, -1)
        cv2.putText(frame, "Output", (proc_px + 10, proc_py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mode_color, 1)
        
        # Draw trajectory circle
        cv2.circle(frame, (center_x, center_y), int(0.5 * scale), (100, 100, 100), 1)
        
        # Show active filters count
        active_filters = len(kalman_tracker.filters)
        cv2.putText(frame, f"Active Filters: {active_filters}", (frame_width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Kalman Filter Demo", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('f') or key == ord('F'):
            filtering_enabled = not filtering_enabled
            status = "ENABLED" if filtering_enabled else "DISABLED"
            print(f"Kalman filtering {status}")
        elif key == ord('r') or key == ord('R'):
            kalman_tracker.filters.clear()
            print("Filter RESET")
        elif key == ord(' '):  # Space
            noise_burst = True
            print("Noise burst added")
        
        # Update time
        t += dt
        time.sleep(dt)
    
    cv2.destroyAllWindows()
    print("Demo completed!")

if __name__ == "__main__":
    main() 
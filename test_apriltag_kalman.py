#!/usr/bin/env python
"""
Test script for AprilTag Kalman Filter
Demonstrates filtering with simulated noisy pose data
"""

import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import AprilTagKalmanFilter
from scipy.spatial.transform import Rotation as R

def simulate_apriltag_motion(num_frames=100, dt=1.0/30.0):
    """
    Simulate AprilTag motion with noise
    Returns ground truth and noisy measurements
    """
    
    # Simulate circular motion for position
    t = np.linspace(0, 2*np.pi, num_frames)
    radius = 0.5
    
    # Ground truth positions
    true_positions = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        2.0 + 0.2 * np.sin(2*t)  # Slight up-down motion
    ])
    
    # Ground truth orientations (rotating around Z axis)
    true_quaternions = []
    for angle in t:
        r = R.from_euler('z', angle)
        quat = r.as_quat()  # [x, y, z, w]
        # Convert to [w, x, y, z] format
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
        true_quaternions.append(quat_wxyz)
    
    true_quaternions = np.array(true_quaternions)
    
    # Add noise to measurements
    position_noise_std = 0.02  # 2cm standard deviation
    orientation_noise_std = 0.05  # Small rotation noise
    
    noisy_positions = true_positions + np.random.normal(0, position_noise_std, true_positions.shape)
    
    noisy_quaternions = []
    for true_quat in true_quaternions:
        # Add small random rotation
        noise_euler = np.random.normal(0, orientation_noise_std, 3)
        noise_rot = R.from_euler('xyz', noise_euler)
        
        true_rot = R.from_quat([true_quat[1], true_quat[2], true_quat[3], true_quat[0]])  # Convert back to [x,y,z,w]
        noisy_rot = true_rot * noise_rot
        
        noisy_quat = noisy_rot.as_quat()  # [x, y, z, w]
        noisy_quat_wxyz = np.array([noisy_quat[3], noisy_quat[0], noisy_quat[1], noisy_quat[2]])
        noisy_quaternions.append(noisy_quat_wxyz)
    
    noisy_quaternions = np.array(noisy_quaternions)
    
    return true_positions, true_quaternions, noisy_positions, noisy_quaternions

def test_position_filtering():
    """Test Kalman filtering for position tracking"""
    print("Testing position + orientation filtering...")
    
    # Generate test data
    true_pos, true_quat, noisy_pos, noisy_quat = simulate_apriltag_motion(100)
    
    # Initialize Kalman filter
    kalman_tracker = AprilTagKalmanFilter(use_position=True, dt=1.0/30.0)
    
    filtered_states = []
    tag_id = 0
    
    for i in range(len(noisy_pos)):
        # Create measurement [x, y, z, q_w, q_x, q_y, q_z]
        measurement = np.concatenate([noisy_pos[i], noisy_quat[i]])
        
        # Update filter
        filtered_state = kalman_tracker.update(tag_id, measurement)
        filtered_states.append(filtered_state)
    
    filtered_states = np.array(filtered_states)
    filtered_positions = filtered_states[:, :3]
    filtered_quaternions = filtered_states[:, 3:7]
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position plots
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        ax = axes[0, 0] if i < 2 else axes[0, 1]
        if i == 2:
            ax = axes[0, 1]
        
        if i < 2:
            ax.plot(true_pos[:, i], label=f'True {axis_name}', color='green', linewidth=2)
            ax.plot(noisy_pos[:, i], label=f'Noisy {axis_name}', color='red', alpha=0.7, linestyle='--')
            ax.plot(filtered_positions[:, i], label=f'Filtered {axis_name}', color='blue', linewidth=2)
            ax.set_title(f'Position {axis_name}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Position (m)')
            ax.legend()
            ax.grid(True)
    
    # Z position in separate subplot
    axes[0, 1].plot(true_pos[:, 2], label='True Z', color='green', linewidth=2)
    axes[0, 1].plot(noisy_pos[:, 2], label='Noisy Z', color='red', alpha=0.7, linestyle='--')
    axes[0, 1].plot(filtered_positions[:, 2], label='Filtered Z', color='blue', linewidth=2)
    axes[0, 1].set_title('Position Z')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Quaternion w component
    axes[1, 0].plot(true_quat[:, 0], label='True Q_w', color='green', linewidth=2)
    axes[1, 0].plot(noisy_quat[:, 0], label='Noisy Q_w', color='red', alpha=0.7, linestyle='--')
    axes[1, 0].plot(filtered_quaternions[:, 0], label='Filtered Q_w', color='blue', linewidth=2)
    axes[1, 0].set_title('Quaternion W Component')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Q_w')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Position error
    position_error_noisy = np.linalg.norm(noisy_pos - true_pos, axis=1)
    position_error_filtered = np.linalg.norm(filtered_positions - true_pos, axis=1)
    
    axes[1, 1].plot(position_error_noisy, label='Noisy Error', color='red', alpha=0.7)
    axes[1, 1].plot(position_error_filtered, label='Filtered Error', color='blue', linewidth=2)
    axes[1, 1].set_title('Position Error')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Error (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('kalman_filter_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    avg_error_noisy = np.mean(position_error_noisy)
    avg_error_filtered = np.mean(position_error_filtered)
    improvement = (avg_error_noisy - avg_error_filtered) / avg_error_noisy * 100
    
    print(f"Average position error (noisy): {avg_error_noisy:.4f} m")
    print(f"Average position error (filtered): {avg_error_filtered:.4f} m")
    print(f"Improvement: {improvement:.1f}%")

def test_orientation_only():
    """Test Kalman filtering for orientation only"""
    print("\nTesting orientation-only filtering...")
    
    # Generate test data
    _, true_quat, _, noisy_quat = simulate_apriltag_motion(50)
    
    # Initialize Kalman filter (orientation only)
    kalman_tracker = AprilTagKalmanFilter(use_position=False, dt=1.0/30.0)
    
    filtered_quaternions = []
    tag_id = 0
    
    for i in range(len(noisy_quat)):
        # Update filter with quaternion only
        filtered_quat = kalman_tracker.update(tag_id, noisy_quat[i])
        filtered_quaternions.append(filtered_quat)
    
    filtered_quaternions = np.array(filtered_quaternions)
    
    # Plot quaternion components
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    quat_labels = ['Q_w', 'Q_x', 'Q_y', 'Q_z']
    
    for i in range(4):
        ax = axes[i//2, i%2]
        ax.plot(true_quat[:, i], label=f'True {quat_labels[i]}', color='green', linewidth=2)
        ax.plot(noisy_quat[:, i], label=f'Noisy {quat_labels[i]}', color='red', alpha=0.7, linestyle='--')
        ax.plot(filtered_quaternions[:, i], label=f'Filtered {quat_labels[i]}', color='blue', linewidth=2)
        ax.set_title(f'Quaternion Component {quat_labels[i]}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('orientation_filter_test.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_missing_measurements():
    """Test behavior when measurements are missing"""
    print("\nTesting missing measurement handling...")
    
    # Generate test data
    true_pos, true_quat, noisy_pos, noisy_quat = simulate_apriltag_motion(100)
    
    # Initialize Kalman filter
    kalman_tracker = AprilTagKalmanFilter(use_position=True, dt=1.0/30.0)
    
    filtered_states = []
    tag_id = 0
    
    for i in range(len(noisy_pos)):
        # Simulate missing measurements (skip every 5th frame)
        if i % 10 == 0:
            # Missing measurement - prediction only
            predicted_state = kalman_tracker.predict_only(tag_id)
            if predicted_state is not None:
                filtered_states.append(predicted_state)
            else:
                # First frame or filter doesn't exist
                measurement = np.concatenate([noisy_pos[i], noisy_quat[i]])
                filtered_state = kalman_tracker.update(tag_id, measurement)
                filtered_states.append(filtered_state)
        else:
            # Normal measurement update
            measurement = np.concatenate([noisy_pos[i], noisy_quat[i]])
            filtered_state = kalman_tracker.update(tag_id, measurement)
            filtered_states.append(filtered_state)
    
    filtered_states = np.array(filtered_states)
    filtered_positions = filtered_states[:, :3]
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        plt.subplot(1, 3, i+1)
        plt.plot(true_pos[:, i], label=f'True {axis_name}', color='green', linewidth=2)
        
        # Mark missing measurements
        measurement_indices = [j for j in range(len(noisy_pos)) if j % 10 != 0]
        missing_indices = [j for j in range(len(noisy_pos)) if j % 10 == 0]
        
        plt.scatter(measurement_indices, noisy_pos[measurement_indices, i], 
                   label=f'Measurements {axis_name}', color='red', alpha=0.7, s=10)
        plt.scatter(missing_indices, noisy_pos[missing_indices, i], 
                   label=f'Missing {axis_name}', color='orange', marker='x', s=30)
        
        plt.plot(filtered_positions[:, i], label=f'Filtered {axis_name}', color='blue', linewidth=2)
        plt.title(f'Position {axis_name} with Missing Measurements')
        plt.xlabel('Frame')
        plt.ylabel('Position (m)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('missing_measurements_test.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("AprilTag Kalman Filter Test Suite")
    print("=" * 40)
    
    # Run tests
    test_position_filtering()
    test_orientation_only()
    test_missing_measurements()
    
    print("\nAll tests completed! Check the generated PNG files for results.") 
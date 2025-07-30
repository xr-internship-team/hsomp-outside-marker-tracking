import numpy as np
from numpy.linalg import inv as inverse

"""
Kalman Filter for AprilTag Pose Estimation
Adapted from quaternion-based attitude estimation

State vector: 7D [x, y, z, q_w, q_x, q_y, q_z] (translation + quaternion)
OR 4D [q_w, q_x, q_y, q_z] (quaternion only)

x -> state estimate;
z -> state measurement;
F -> state-transition model;
H -> observation model;
P -> process covariance;
Q -> covariance of the process noise;
R -> covariance of the observation noise;
K -> kalman gain;
"""

class KalmanFilter:
    def __init__(self, x0, F, H, P, Q, R):
        self.n = F.shape[1]  # state dimension
        self.m = H.shape[0]  # measurement dimension
        
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.P = P  # Covariance matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state
        
    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def correct(self, z):
        """Correction step with measurement z"""
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        self.K = self.P @ self.H.T @ inverse(S)
        
        # Update state
        self.x = self.x + self.K @ y
        
        # Update covariance
        I = np.eye(self.n)
        self.P = (I - self.K @ self.H) @ self.P
        
        return self.x
    
    def update_state_transition(self, F):
        """Update state transition matrix (for time-varying systems)"""
        self.F = F
    
    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()


class AprilTagKalmanFilter:
    """Specialized Kalman filter for AprilTag pose tracking"""
    
    def __init__(self, use_position=True, dt=1.0/30.0):
        """
        Initialize AprilTag Kalman filter
        
        Args:
            use_position: If True, track position+orientation (7D), else just orientation (4D)
            dt: Time step between measurements
        """
        self.use_position = use_position
        self.dt = dt
        self.filters = {}  # Dictionary to store filters for different tag IDs
        
        if use_position:
            self.state_dim = 7  # [x, y, z, q_w, q_x, q_y, q_z]
            self.meas_dim = 7   # Same as state for direct measurement
        else:
            self.state_dim = 4  # [q_w, q_x, q_y, q_z]
            self.meas_dim = 4   # Same as state for direct measurement
            
    def _create_filter(self, initial_measurement):
        """Create a new Kalman filter for a tag"""
        
        if self.use_position:
            # 7D state: [x, y, z, q_w, q_x, q_y, q_z]
            x0 = initial_measurement.reshape(-1, 1)
            
            # State transition matrix (constant position/orientation model)
            F = np.eye(7)
            
            # Observation matrix (direct measurement of state)
            H = np.eye(7)
            
            # Initial covariance
            P = np.eye(7) * 1.0
            
            # Process noise (small uncertainty in prediction)
            Q = np.eye(7) * 1e-4
            Q[3:, 3:] *= 10  # Higher uncertainty for quaternion
            
            # Measurement noise (AprilTag detection uncertainty)
            R = np.eye(7) * 0.01
            R[:3, :3] *= 10  # Higher uncertainty for position
            
        else:
            # 4D state: [q_w, q_x, q_y, q_z]
            x0 = initial_measurement.reshape(-1, 1)
            
            # State transition matrix (constant orientation model)
            F = np.eye(4)
            
            # Observation matrix (direct measurement of quaternion)
            H = np.eye(4)
            
            # Initial covariance
            P = np.eye(4) * 1.0
            
            # Process noise
            Q = np.eye(4) * 1e-4
            
            # Measurement noise
            R = np.eye(4) * 0.01
            
        return KalmanFilter(x0, F, H, P, Q, R)
    
    def update(self, tag_id, measurement):
        """
        Update filter for a specific tag
        
        Args:
            tag_id: AprilTag ID
            measurement: numpy array [x,y,z,q_w,q_x,q_y,q_z] or [q_w,q_x,q_y,q_z]
        
        Returns:
            Filtered state estimate
        """
        if tag_id not in self.filters:
            # Create new filter for this tag
            self.filters[tag_id] = self._create_filter(measurement)
            return measurement
        
        # Get existing filter
        kf = self.filters[tag_id]
        
        # Predict
        kf.predict()
        
        # Correct with measurement
        filtered_state = kf.correct(measurement.reshape(-1, 1))
        
        # Normalize quaternion part
        if self.use_position:
            # Normalize quaternion (last 4 elements)
            quat = filtered_state[3:7].flatten()
            quat_norm = quat / np.linalg.norm(quat)
            filtered_state[3:7] = quat_norm.reshape(-1, 1)
        else:
            # Normalize entire state (quaternion only)
            quat = filtered_state.flatten()
            quat_norm = quat / np.linalg.norm(quat)
            filtered_state = quat_norm.reshape(-1, 1)
            
        # Update filter state with normalized quaternion
        kf.x = filtered_state
        
        return filtered_state.flatten()
    
    def predict_only(self, tag_id):
        """
        Prediction step only (when tag is not detected)
        
        Args:
            tag_id: AprilTag ID
        
        Returns:
            Predicted state or None if filter doesn't exist
        """
        if tag_id not in self.filters:
            return None
            
        kf = self.filters[tag_id]
        kf.predict()
        
        # Normalize quaternion part
        if self.use_position:
            quat = kf.x[3:7].flatten()
            quat_norm = quat / np.linalg.norm(quat)
            kf.x[3:7] = quat_norm.reshape(-1, 1)
        else:
            quat = kf.x.flatten()
            quat_norm = quat / np.linalg.norm(quat)
            kf.x = quat_norm.reshape(-1, 1)
            
        return kf.x.flatten()
    
    def get_state(self, tag_id):
        """Get current state estimate for a tag"""
        if tag_id in self.filters:
            return self.filters[tag_id].get_state().flatten()
        return None
    
    def remove_filter(self, tag_id):
        """Remove filter for a tag (e.g., when tag is lost)"""
        if tag_id in self.filters:
            del self.filters[tag_id] 
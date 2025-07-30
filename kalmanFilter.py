from filterpy.kalman import KalmanFilter
import numpy as np

class PoseKalmanFilter:
    def __init__(self, initial_position, initial_orientation, 
                         process_noise=0.1, measurement_noise=0.3,
                         orientation_filter_gain=0.2, dt=0.333):
        # Store basic parameters
        self.base_measurement_noise = measurement_noise
        self.base_process_noise = process_noise
        self.dt = dt
        
        # Initialize position filter with anti-drift mechanisms
        self.pos_filter = self._init_position_filter(initial_position, dt, process_noise, measurement_noise)
        
        # Orientation handling
        self.orientation = self._normalize_quaternion(np.array(initial_orientation))
        self.orientation_filter_gain = orientation_filter_gain
        self.last_valid_orientation = self.orientation.copy()
        
        # 🛡️ ANTI-DRIFT MECHANISMS 🛡️
        
        # 1. Confidence tracking with CAPPED R values
        self.smoothed_confidence = 0.5  # Start with medium confidence
        self.confidence_alpha = 0.3  # Smoothing factor
        self.max_decision_margin = 20.0
        self.min_r_scale = 0.1  # Minimum R scaling
        self.max_r_scale = 3.0   # 🔧 REDUCED from 10.0 to 3.0 to prevent over-distrust
        
        # 2. P Matrix protection - Prevent filter freeze
        self.min_p_position = 0.01    # Minimum P for position states
        self.min_p_velocity = 0.001   # Minimum P for velocity states  
        self.p_boost_interval = 100   # Every N updates, boost P
        self.p_boost_factor = 1.1     # Boost factor
        
        # 3. Drift monitoring and correction
        self.position_history = []    # Track position history
        self.drift_window = 20        # Window for drift detection
        self.max_z_drift_rate = 0.001 # Max allowed Z-axis drift per update
        self.drift_correction_factor = 0.9  # Correction strength
        
        # 4. Health monitoring
        self.update_count = 0
        self.low_confidence_count = 0
        self.high_r_count = 0
        self.filter_health_score = 1.0
        
        # 5. Adaptive process noise injection
        self.adaptive_q_enabled = True
        self.q_boost_on_low_confidence = True
        self.q_boost_factor = 2.0
    
    def _init_position_filter(self, initial_position, dt, process_noise, measurement_noise):
        """Initialize Kalman filter with anti-drift protections"""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0])
        
        # Transition matrix
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise matrix
        kf.Q = np.eye(6) * process_noise
        
        # Measurement noise matrix
        kf.R = np.eye(3) * measurement_noise
        
        # 🛡️ LARGER initial P to prevent premature convergence
        kf.P = np.eye(6) * 1000  # Increased from 500 to 1000
        
        return kf
    
    def _protect_p_matrix(self):
        """Prevent P matrix from becoming too small (filter freeze)"""
        # Enforce minimum P values
        for i in range(3):  # Position states
            if self.pos_filter.P[i, i] < self.min_p_position:
                self.pos_filter.P[i, i] = self.min_p_position
        
        for i in range(3, 6):  # Velocity states
            if self.pos_filter.P[i, i] < self.min_p_velocity:
                self.pos_filter.P[i, i] = self.min_p_velocity
        
        # Periodic P boost to prevent over-convergence
        if self.update_count % self.p_boost_interval == 0 and self.update_count > 0:
            self.pos_filter.P *= self.p_boost_factor
            # print(f"P matrix boosted at update {self.update_count}")
    
    def _detect_and_correct_drift(self, current_position):
        """Monitor and correct systematic drift, especially Z-axis"""
        self.position_history.append(current_position.copy())
        
        # Keep only recent history
        if len(self.position_history) > self.drift_window:
            self.position_history.pop(0)
        
        # Check for Z-axis drift if we have enough history
        if len(self.position_history) >= self.drift_window:
            positions = np.array(self.position_history)
            
            # Calculate Z-axis drift rate
            z_positions = positions[:, 2]
            z_drift_rate = (z_positions[-1] - z_positions[0]) / len(z_positions)
            
            # If excessive downward drift detected
            if z_drift_rate < -self.max_z_drift_rate:
                # Apply gentle upward correction
                correction = -z_drift_rate * self.drift_correction_factor
                self.pos_filter.x[2] += correction
                print(f"⚠️  Z-drift correction applied: {correction:.4f}m")
    
    def _monitor_filter_health(self, confidence, r_scale):
        """Monitor filter health and detect problematic behavior"""
        # Track concerning patterns
        if confidence is not None and confidence < 0.3:
            self.low_confidence_count += 1
        
        if r_scale is not None and r_scale > 2.0:
            self.high_r_count += 1
        
        # Calculate health score
        total_updates = max(self.update_count, 1)
        low_conf_ratio = self.low_confidence_count / total_updates
        high_r_ratio = self.high_r_count / total_updates
        
        self.filter_health_score = 1.0 - 0.5 * (low_conf_ratio + high_r_ratio)
        
        # Auto-recovery if health is poor
        if self.filter_health_score < 0.5 and self.update_count > 50:
            self._emergency_filter_reset()
    
    def _emergency_filter_reset(self):
        """Emergency filter reset if health deteriorates"""
        print("🚨 Emergency filter reset triggered!")
        
        # Reset P matrix to larger values
        self.pos_filter.P = np.eye(6) * 1000
        
        # Reset confidence tracking
        self.smoothed_confidence = 0.5
        
        # Reset health counters
        self.low_confidence_count = 0
        self.high_r_count = 0
        self.filter_health_score = 1.0
    
    def _adaptive_process_noise(self, confidence):
        """Inject adaptive process noise to keep filter humble"""
        if not self.adaptive_q_enabled:
            return
        
        base_Q = np.eye(6) * self.base_process_noise
        
        # Boost Q when confidence is low to keep filter flexible
        if confidence is not None and confidence < 0.4 and self.q_boost_on_low_confidence:
            boost_factor = self.q_boost_factor * (1.0 - confidence)
            self.pos_filter.Q = base_Q * boost_factor
        else:
            self.pos_filter.Q = base_Q
    
    def update_confidence_and_r_matrix(self, decision_margin):
        """Update R matrix with BOUNDED scaling to prevent over-distrust"""
        # Normalize decision_margin
        normalized_conf = np.clip(decision_margin / self.max_decision_margin, 0.0, 1.0)
        
        # Smooth confidence
        self.smoothed_confidence = (self.confidence_alpha * normalized_conf + 
                                   (1 - self.confidence_alpha) * self.smoothed_confidence)
        
        # 🔧 BOUNDED R scaling - prevents over-distrust
        r_scale = self.max_r_scale - (self.max_r_scale - self.min_r_scale) * self.smoothed_confidence
        r_scale = np.clip(r_scale, self.min_r_scale, self.max_r_scale)  # Double-check bounds
        
        # Update R matrix
        base_R = np.eye(3) * self.base_measurement_noise
        self.pos_filter.R = base_R * r_scale
        
        return self.smoothed_confidence, r_scale
    
    # ... existing quaternion methods remain the same ...
    def _normalize_quaternion(self, q):
        """Quaternion normalizasyonu ve validasyonu"""
        q = np.array(q)
        if np.any(np.isnan(q)):
            return self.last_valid_orientation
        
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            return self.last_valid_orientation
            
        q_normalized = q / norm
        self.last_valid_orientation = q_normalized
        return q_normalized
    
    def _safe_slerp(self, q1, q2, t):
        """NaN'lardan korunmalı SLERP implementasyonu"""
        try:
            q1 = self._normalize_quaternion(q1)
            q2 = self._normalize_quaternion(q2)
            
            dot = np.dot(q1, q2)
            dot = np.clip(dot, -1.0, 1.0)
            
            theta = np.arccos(dot) * t
            rel_rot = q2 - q1 * dot
            rel_rot_norm = np.linalg.norm(rel_rot)
            
            if rel_rot_norm < 1e-6:
                return q1
                
            rel_rot = rel_rot / rel_rot_norm
            result = q1 * np.cos(theta) + rel_rot * np.sin(theta)
            return self._normalize_quaternion(result)
            
        except:
            return self.last_valid_orientation
    
    def update(self, position, orientation, decision_margin=None):
        """Main update with comprehensive anti-drift protection"""
        self.update_count += 1
        
        # 1. Update confidence and R matrix if available
        current_confidence = None
        r_scale = None
        if decision_margin is not None:
            current_confidence, r_scale = self.update_confidence_and_r_matrix(decision_margin)
        
        # 2. Adaptive process noise injection
        self._adaptive_process_noise(current_confidence)
        
        # 3. Standard Kalman prediction
        self.pos_filter.predict()
        
        # 4. Measurement update
        self.pos_filter.update(position)
        
        # 5. P matrix protection (prevent filter freeze)
        self._protect_p_matrix()
        
        # 6. Drift detection and correction
        self._detect_and_correct_drift(self.pos_filter.x[:3])
        
        # 7. Filter health monitoring
        self._monitor_filter_health(current_confidence, r_scale)
        
        # 8. Orientation update
        try:
            orientation = np.array(orientation)
            if np.any(np.isnan(orientation)):
                orientation = self.last_valid_orientation
                
            self.orientation = self._safe_slerp(self.orientation, orientation, self.orientation_filter_gain)
        except:
            pass
        
        return self.pos_filter.x[:3], self.orientation, current_confidence, r_scale
    
    def get_filter_diagnostics(self):
        """Get comprehensive filter health diagnostics"""
        return {
            'health_score': self.filter_health_score,
            'update_count': self.update_count,
            'low_confidence_ratio': self.low_confidence_count / max(self.update_count, 1),
            'high_r_ratio': self.high_r_count / max(self.update_count, 1),
            'p_trace': np.trace(self.pos_filter.P),
            'smoothed_confidence': self.smoothed_confidence,
            'current_r_scale': np.mean(np.diag(self.pos_filter.R)) / self.base_measurement_noise
        }
    
    def set_confidence_params(self, max_decision_margin=None, min_r_scale=None, max_r_scale=None, alpha=None):
        """Tune confidence mapping parameters"""
        if max_decision_margin is not None:
            self.max_decision_margin = max_decision_margin
        if min_r_scale is not None:
            self.min_r_scale = min_r_scale
        if max_r_scale is not None:
            self.max_r_scale = max_r_scale  # Still capped at 3.0 in bounds check
        if alpha is not None:
            self.confidence_alpha = alpha
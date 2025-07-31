from filterpy.kalman import KalmanFilter
import numpy as np

class PoseKalmanFilter:
    def __init__(self, initial_position, initial_orientation, 
                         process_noise=0.1, measurement_noise=0.3,
                         orientation_filter_gain=0.2, dt=0.333):
        # Pozisyon filtresi
        self.pos_filter = self._init_position_filter(initial_position, dt, process_noise, measurement_noise)
        
        # Oryantasyon için başlangıç değerleri ve parametreler
        self.orientation = self._normalize_quaternion(np.array(initial_orientation))
        self.orientation_filter_gain = orientation_filter_gain
        self.last_valid_orientation = self.orientation.copy()
    
    def _init_position_filter(self, initial_position, dt, process_noise, measurement_noise):
        """Pozisyon Kalman filtresi başlatma"""
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0])
        
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        kf.Q = np.eye(6) * process_noise
        kf.R = np.eye(3) * measurement_noise
        kf.P = np.eye(6) * 500
        
        return kf
    
    def _normalize_quaternion(self, q):
        """Quaternion normalizasyonu ve validasyonu"""
        q = np.array(q)
        if np.any(np.isnan(q)):
            return self.last_valid_orientation
        
        norm = np.linalg.norm(q)
        if norm < 1e-6:  # Çok küçük norm
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
    
    def update(self, position, orientation):
        # Pozisyon güncelleme
        self.pos_filter.predict()
        self.pos_filter.update(position)
        
        # Oryantasyon güncelleme
        try:
            orientation = np.array(orientation)
            if np.any(np.isnan(orientation)):
                orientation = self.last_valid_orientation
                
            self.orientation = self._safe_slerp(self.orientation, orientation, self.orientation_filter_gain)
        except:
            pass  # Hata durumunda son geçerli oryantasyonu koru
        
        return self.pos_filter.x[:3], self.orientation
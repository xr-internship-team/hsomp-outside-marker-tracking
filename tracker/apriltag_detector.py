# tracker/apriltag_detector.py
import numpy as np, cv2
from pupil_apriltags import Detector
from .transforms import rt_to_T
from itertools import combinations

class AprilTagDetector:
    def __init__(self, K, D, family="tag25h9"):
        self.K = K; self.D = D
        self.detector = Detector(
            families=family, nthreads=1, quad_decimate=1,
            quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0
        )
        self._last_raw = None
        self._detections = {}

    @staticmethod
    def _object_points(size_m):
        s = float(size_m)/2.0
        # TL, TR, BR, BL (pupil_apriltags sırası)
        return np.array([
            [-s, -s, 0.0],
            [ +s, -s, 0.0],
            [ +s, +s, 0.0],
            [ -s, +s, 0.0],
        ], dtype=np.float64)

    def _solve_pose(self, corners, size_m):
        obj = self._object_points(size_m)
        img = np.asarray(corners, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D)
        Rm, _ = cv2.Rodrigues(rvec)
        return Rm, tvec.reshape(3), rvec

    def detect(self, gray, tag_sizes):
        # tek pass detect (dummy size). Poza kendimiz solvePnP yapacağız
        raw = self.detector.detect(
            gray, estimate_tag_pose=False,
            camera_params=[self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]],
            tag_size=0.08
        )
        self._last_raw = raw
        dets = {}
        for d in raw:
            tid = int(d.tag_id)
            size = float(tag_sizes.get(tid, 0.08))
            Rm, tvec, rvec = self._solve_pose(d.corners, size)
            T_c_t = rt_to_T(Rm, tvec)
            dets[tid] = {
                "T_c_t": T_c_t,
                "dm": float(d.decision_margin),
                "center": tuple(map(int, d.center)),
                "rvec": rvec, "tvec": tvec, "rmat": Rm,
            }
        self._detections = dets
        return dets

    def update_graph_pairs(self, graph):
        tids = list(self._detections.keys())
        for i, j in combinations(tids, 2):
            Ti = self._detections[i]["T_c_t"]
            Tj = self._detections[j]["T_c_t"]
            graph.update_pair(i, Ti, j, Tj)

    def error_vs_front(self, front_id, fused_pos, fused_quat):
        """Ön görünüyorsa front-head ile fused farkını döndür (deg, m)."""
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        if front_id not in self._detections: return None, None
        Rf = self._detections[front_id]["rmat"]
        tf = self._detections[front_id]["tvec"]
        qf = R.from_matrix(Rf).as_quat()
        Rdiff = R.from_quat(qf).inv() * R.from_quat(fused_quat)
        ang = np.degrees(2*np.arccos(np.clip(Rdiff.as_quat()[-1], -1, 1)))
        pos = np.linalg.norm(tf - fused_pos)
        return ang, pos

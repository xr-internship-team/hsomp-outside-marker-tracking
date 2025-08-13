# tracker/detector_ptag.py
import numpy as np
from pupil_apriltags import Detector
from .transforms import project_to_SO3
class PTagDetector:
    def __init__(self, K, D, family="tag25h9"):
        self.K, self.D = K, D
        self.fx, self.fy, self.cx, self.cy = K[0,0], K[1,1], K[0,2], K[1,2]
        self.detector = Detector(
            families=family, nthreads=1,
            quad_decimate=1, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.25, debug=0
        )

    def detect_with_per_id_size(self, gray, tag_sizes: dict):
        """
        Senin akışın:
        1) Once default tag_size ile detect (pose'lu)
        2) Her ID için gerçek boyla tekrar detect edip pozları al
        """
        base = self.detector.detect(
            gray, estimate_tag_pose=True,
            camera_params=[self.fx, self.fy, self.cx, self.cy],
            tag_size=0.08
        )
        out = []
        for d in base:
            tid = int(d.tag_id)
            tsz = float(tag_sizes.get(tid, 0.08))
            det_list = self.detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=[self.fx, self.fy, self.cx, self.cy],
                tag_size=tsz
            )
            dd = next((t for t in det_list if int(t.tag_id) == tid), None)
            if dd is None:
                continue
            rmat = project_to_SO3(dd.pose_R)
            tvec = dd.pose_t.reshape(3)
            out.append({
                "id": tid,
                "rmat": rmat,
                "tvec": tvec,
                "dm": float(dd.decision_margin),
                "center": tuple(map(int, dd.center))
            })
        return out

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
       - Eğer tüm etiketler aynı boyda ise: TEK detect çağrısı.
       - Eğer farklı boylar varsa: benzersiz boy başına TEK detect çağrısı.
       Sonuçlar tek listede birleştirilir.
       """
       out = []
       if not tag_sizes:
           return out
       # Boy -> bu boyu kullanan ID'ler
       size_to_ids = {}
       for tid, sz in tag_sizes.items():
           size_to_ids.setdefault(float(sz), set()).add(int(tid))

       def _run_detect(tag_size):
           return self.detector.detect(
               gray, estimate_tag_pose=True,
               camera_params=(self.fx, self.fy, self.cx, self.cy),
               tag_size=float(tag_size)
           )

       unique_sizes = list(size_to_ids.keys())
       if len(unique_sizes) == 1:
           det_list = _run_detect(unique_sizes[0])
           for d in det_list:
               tid = int(d.tag_id)
               if tid not in size_to_ids[unique_sizes[0]]:
                   continue
               rmat = project_to_SO3(d.pose_R)
               tvec = d.pose_t.reshape(3)
               out.append({
                   "id": tid,
                   "rmat": rmat,
                   "tvec": tvec,
                   "dm": float(d.decision_margin),
                   "center": tuple(map(int, d.center))
               })
       else:
           for sz, valid_ids in size_to_ids.items():
               det_list = _run_detect(sz)
               for d in det_list:
                   tid = int(d.tag_id)
                   if tid not in valid_ids:
                       continue
                   rmat = project_to_SO3(d.pose_R)
                   tvec = d.pose_t.reshape(3)
                   out.append({
                       "id": tid,
                       "rmat": rmat,
                       "tvec": tvec,
                       "dm": float(d.decision_margin),
                       "center": tuple(map(int, d.center))
                   })
       return out
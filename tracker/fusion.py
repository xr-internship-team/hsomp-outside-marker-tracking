import numpy as np
from scipy.spatial.transform import Rotation as R
from .transforms import invert_T, decompose_T, avg_quaternions
from kalmanFilter import PoseKalmanFilter
from config import DM_MIN, WEIGHT_BETA

_global_filter = None
# kareden kareye ağırlıkları yumuşatmak için
_prev_w = {}

def fuse_head_pose(detections, graph, use_filter=True):
   """
   - DM kapısı: düşük karar marjlı ölçümler füzyona girmez
   - Ağırlık EMA: marker geçişinde ani sıçramayı yumuşatır
   """
   global _global_filter, _prev_w
   head_positions, head_quats, w_raw_list, ids = [], [], [], []

   for tid, info in detections.items():
       # 1) görünür ve güvenilir mi?
       if float(info.get("dm", 0.0)) < DM_MIN:
           continue
       # 2) graf üzerinden head pozu için gerekli dönüşüm
       T_h_t = graph.T_root_to(tid)  # tag->head (root=head)
       if T_h_t is None:
           continue
       T_c_t = info["T_c_t"]
       T_c_h = T_c_t @ invert_T(T_h_t)
       R_ch, t_ch = decompose_T(T_c_h)
       head_positions.append(t_ch)
       head_quats.append(R.from_matrix(R_ch).as_quat())
       w_raw_list.append(max(float(info["dm"]), 0.0))
       ids.append(tid)

   if not ids:
       return None

   # --- Ağırlık EMA (histerezis) ---
   w_ema = []
   for tid, w_raw in zip(ids, w_raw_list):
       w_prev = _prev_w.get(tid, w_raw)
       w_now  = WEIGHT_BETA * w_prev + (1.0 - WEIGHT_BETA) * w_raw
       _prev_w[tid] = w_now
       w_ema.append(w_now)
   w = np.asarray(w_ema, dtype=np.float64)
   if np.allclose(w.sum(), 0.0):
       w = np.ones_like(w)
   w = w / w.sum()

   fused_pos  = np.average(np.vstack(head_positions), axis=0, weights=w)
   fused_quat = avg_quaternions(head_quats, w)
   avg_dm = float(np.average(np.asarray([detections[i]["dm"] for i in ids], dtype=float), weights=w))

   if use_filter:
       if _global_filter is None:
           _global_filter = PoseKalmanFilter(fused_pos, fused_quat)
       fused_pos, fused_quat, confidence, r_scale = _global_filter.update(fused_pos, fused_quat, avg_dm)
   else:
       confidence, r_scale = avg_dm, None

   return fused_pos, fused_quat, confidence, r_scale, ids, avg_dm

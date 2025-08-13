# tracker/fusion.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from .transforms import invert_T, decompose_T, avg_quaternions, project_to_SO3
from kalmanFilter import PoseKalmanFilter

_global_filter = None

def fuse_head_pose(detections, graph, use_filter=True):
    global _global_filter
    head_positions, head_quats, weights, used_ids = [], [], [], []

    for tid, info in detections.items():
        T_h_t = graph.T_root_to(tid)  # tag->head (head=root)
        if T_h_t is None: continue

        T_c_t = info["T_c_t"]
        T_c_h = T_c_t @ invert_T(T_h_t)

        # --- Güvenli decompose/quat ---
        R_c_h, t_c_h = decompose_T(T_c_h)            # burada already project_to_SO3 var
        R_c_h = project_to_SO3(R_c_h)                # çift güvenlik (isteğe bağlı)
        q_c_h = R.from_matrix(R_c_h).as_quat()

        head_positions.append(t_c_h)
        head_quats.append(q_c_h)
        weights.append(max(info["dm"], 0.0))
        used_ids.append(tid)

    if not head_quats:
        return None

    w = np.array(weights, dtype=float)
    if w.sum() == 0: w = np.ones_like(w)
    w = w / w.sum()

    fused_pos  = np.average(np.vstack(head_positions), axis=0, weights=w)
    fused_quat = avg_quaternions(head_quats, w)
    avg_dm = float(np.average([detections[i]["dm"] for i in used_ids], weights=w))

    if use_filter:
        if _global_filter is None:
            _global_filter = PoseKalmanFilter(fused_pos, fused_quat)
        fused_pos, fused_quat, confidence, r_scale = _global_filter.update(fused_pos, fused_quat, avg_dm)
    else:
        confidence, r_scale = avg_dm, None

    return fused_pos, fused_quat, confidence, r_scale, used_ids, avg_dm

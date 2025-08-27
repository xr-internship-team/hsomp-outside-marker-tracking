# tracker/transforms.py
import numpy as np
from scipy.spatial.transform import Rotation as R

def project_to_SO3(Rm: np.ndarray) -> np.ndarray:
    """
    En yakın ortogonal + det=+1 matrise projeksiyon (SVD).
    """
    U, S, Vt = np.linalg.svd(Rm)
    Rfix = U @ Vt
    if np.linalg.det(Rfix) < 0:
        U[:, -1] *= -1
        Rfix = U @ Vt
    return Rfix

def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = project_to_SO3(np.asarray(Rm, dtype=float))
    T[:3, 3] = np.asarray(t).reshape(3)
    return T

def invert_T(T):
    Rm = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = Rm.T
    Ti[:3, 3]  = -Rm.T @ t
    return Ti

def decompose_T(T):
    # <<< ÖNEMLİ: R'yi her seferinde SO(3)'e zorla
    Rm = project_to_SO3(T[:3,:3])
    t  = T[:3,3].copy()
    return Rm, t

def avg_quaternions(quats, weights):
    M = np.zeros((4,4), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0: w = np.ones_like(w)
    w = w / w.sum()
    for qi, wi in zip(quats, w):
        q = np.asarray(qi, dtype=np.float64).reshape(4,1)
        if q[3,0] < 0: q = -q
        M += wi * (q @ q.T)
    _, V = np.linalg.eigh(M)
    q = V[:, -1]
    return q / np.linalg.norm(q)

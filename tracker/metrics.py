# tracker/metrics.py
import numpy as np
from scipy.spatial.transform import Rotation as R
from .transforms import invert_T, decompose_T

def _rot_err_deg(R1, R2):
    Rdiff = R1.T @ R2
    # trace yöntemi (sayısal kararlı)
    val = (np.trace(Rdiff) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def _pose_err(T_pred, T_meas):
    R_pred, t_pred = decompose_T(T_pred)
    R_meas, t_meas = decompose_T(T_meas)
    ang = _rot_err_deg(R_pred, R_meas)
    pos = float(np.linalg.norm(t_pred - t_meas))
    return ang, pos

def pair_residuals(detections, graph):
    """
    Görünen her (i,j) çifti için:
    T_c_tj_pred = T_c_ti @ (T_i_j_graph)
    ile T_c_tj ölçümünü kıyasla.
    """
    tids = list(detections.keys())
    res = []
    for a in range(len(tids)):
        for b in range(a+1, len(tids)):
            i, j = tids[a], tids[b]
            T_i_j = graph.T_a_to_b(i, j)  # yoksa None
            if T_i_j is None:
                continue
            T_c_ti = detections[i]["T_c_t"]
            T_c_tj_meas = detections[j]["T_c_t"]
            T_c_tj_pred = T_c_ti @ T_i_j
            ang, pos = _pose_err(T_c_tj_pred, T_c_tj_meas)
            res.append(((i, j), ang, pos))
    # en kötüyü en üste
    res.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return res

def cycle_closure_errors(graph, nodes=None, max_list=5):
    """
    Üçlü döngüler (i,j,k) için: T_i_j * T_j_k * T_k_i ≈ I
    E=I^-1 * (çarpım) hatasını açı/poz olarak raporla.
    """
    if nodes is None:
        nodes = graph.reachable_nodes()
    errs = []
    for a in range(len(nodes)):
        for b in range(a+1, len(nodes)):
            for c in range(b+1, len(nodes)):
                i, j, k = nodes[a], nodes[b], nodes[c]
                Tij = graph.T_a_to_b(i, j)
                Tjk = graph.T_a_to_b(j, k)
                Tki = graph.T_a_to_b(k, i)
                if Tij is None or Tjk is None or Tki is None:
                    continue
                T_cycle = Tij @ Tjk @ Tki  # idealde I
                from .transforms import decompose_T
                Rcyc, tcyc = decompose_T(T_cycle)
                ang = _rot_err_deg(Rcyc, np.eye(3))
                pos = float(np.linalg.norm(tcyc))
                errs.append(((i, j, k), ang, pos))
    errs.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return errs[:max_list]

def compute_metrics(detections, graph, front_err=None, max_pairs=1, max_cycles=1):
    """
    Kolay kullanım: en kötü (i,j) artık hatası ve en kötü (i,j,k) döngü hatasını döndür.
    front_err: (ang_deg, pos_m) veya None
    """
    pairs = pair_residuals(detections, graph)
    pair_max = pairs[0] if pairs else None

    cycles = cycle_closure_errors(graph)
    cycle_max = cycles[0] if cycles else None

    return {
        "front_err": front_err,      # (ang, pos) ya da None
        "pair_max": pair_max,        # ((i,j), ang, pos) ya da None
        "cycle_max": cycle_max       # ((i,j,k), ang, pos) ya da None
    }

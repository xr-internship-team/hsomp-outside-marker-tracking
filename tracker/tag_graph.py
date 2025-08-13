# tracker/tag_graph.py
from collections import deque
import json, os, numpy as np
from .transforms import decompose_T, rt_to_T, invert_T, avg_quaternions, project_to_SO3

class TagGraph:
    def __init__(self, root_id, ema_alpha=0.15):
        self.root = int(root_id)
        self.EMA_ALPHA = float(ema_alpha)
        self.edges = {}   # (a,b) -> T_a_b (4x4)
        # kök -> kök = I
        self.edges[(self.root, self.root)] = np.eye(4)

    def _ema_edge(self, a, b, T_new):
        key = (int(a), int(b))
        Rn, tn = decompose_T(T_new)
        # decompose_T zaten SO(3) yapıyor; ama açıkça proje etmek istersen:
        Rn = project_to_SO3(Rn)
        if key not in self.edges:
            self.edges[key] = rt_to_T(Rn, tn)
            return
        Ro, to = decompose_T(self.edges[key])
        from scipy.spatial.transform import Rotation as R
        qo = R.from_matrix(Ro).as_quat()
        qn = R.from_matrix(Rn).as_quat()
        qmix = avg_quaternions([qo, qn], [1-self.EMA_ALPHA, self.EMA_ALPHA])
        Rmix = R.from_quat(qmix).as_matrix()
        tmix = (1-self.EMA_ALPHA)*to + self.EMA_ALPHA*tn
        self.edges[key] = rt_to_T(Rmix, tmix)

    def update_pair(self, i, T_c_ti, j, T_c_tj):
        T_i_j = invert_T(T_c_ti) @ T_c_tj
        self._ema_edge(i, j, T_i_j)
        self._ema_edge(j, i, invert_T(T_i_j))

    def update_with_detections(self, detections):
        tids = list(detections.keys())
        for a in range(len(tids)):
            for b in range(a+1, len(tids)):
                i, j = tids[a], tids[b]
                self.update_pair(i, detections[i]["T_c_t"], j, detections[j]["T_c_t"])

    def T_root_to(self, target_id):
        target_id = int(target_id)
        if target_id == self.root:
            return np.eye(4)
        q = deque([(self.root, np.eye(4))])
        visited = {self.root}
        while q:
            cur, T_root_cur = q.popleft()
            neighs = [b for (a,b) in self.edges.keys() if a == cur]
            for n in neighs:
                if n in visited: continue
                T_cur_n = self.edges[(cur, n)]
                T_root_n = T_root_cur @ T_cur_n
                if n == target_id:
                    return T_root_n
                visited.add(n)
                q.append((n, T_root_n))
        return None

    def reachable_nodes(self):
        out = set([self.root]); q = deque([self.root])
        while q:
            cur = q.popleft()
            neighs = [b for (a,b) in self.edges.keys() if a == cur]
            for n in neighs:
                if n in out: continue
                out.add(n); q.append(n)
        return sorted(out)
    
    def T_a_to_b(self, a, b):
        from .transforms import invert_T
        T_r_a = self.T_root_to(a)
        T_r_b = self.T_root_to(b)
        if T_r_a is None or T_r_b is None: return None
        return invert_T(T_r_a) @ T_r_b



# ---- extrinsics kaydet/yükle köprüsü ----
def dump_graph_as_extrinsics(graph: TagGraph):
    out = {}
    for tid in graph.reachable_nodes():
        T_h_t = graph.T_root_to(tid)    # tag->head (root=head)
        if T_h_t is None: 
            continue
        Rm, t = decompose_T(T_h_t)
        out[str(tid)] = {"R": Rm.tolist(), "t": t.tolist()}
    return out

def seed_graph_from_extrinsics(graph: TagGraph, path_or_dict):
    if isinstance(path_or_dict, str):
        if not os.path.exists(path_or_dict):
            return None
        with open(path_or_dict, "r") as f:
            extr = json.load(f)
    else:
        extr = path_or_dict

    if not extr: 
        return None

    for k, v in extr.items():
        tid = int(k)
        Rm = np.array(v["R"], dtype=float)
        t  = np.array(v["t"], dtype=float)
        T_root_t = rt_to_T(Rm, t)       # tag->root
        # root->tag kenarı = (T_root_t)^-1
        T_r_t = T_root_t
        T_t_r = invert_T(T_r_t)
        graph._ema_edge(graph.root, tid, T_t_r)    # root -> tag
        graph._ema_edge(tid, graph.root, T_r_t)    # tag  -> root
    return extr

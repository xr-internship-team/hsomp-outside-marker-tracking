# tracker/io_extrinsics.py
import json, os, numpy as np
from .transforms import rt_to_T, decompose_T

def save_extrinsics(path, extrinsics_dict):
    with open(path, "w") as f:
        json.dump(extrinsics_dict, f, indent=2)

def load_extrinsics(path):
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        return json.load(f)

def extrinsics_to_json(T_h_t):
    Rm, t = decompose_T(T_h_t)
    return {"R": Rm.tolist(), "t": t.tolist()}

def json_to_extrinsics(d):
    Rm = np.array(d["R"], dtype=float)
    t  = np.array(d["t"], dtype=float)
    return rt_to_T(Rm, t)

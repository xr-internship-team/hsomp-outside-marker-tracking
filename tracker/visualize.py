# tracker/visualize.py
import cv2

def put_hud(frame, fused_pos, used_ids, use_filter, calibrating, avg_dm, ang_err, pos_err, reachable, pair_max =None, cycle_max=None):
    y = 20
    cv2.putText(frame, f'Filter: {"ON" if use_filter else "OFF"} (F)', (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2); y += 20
    cv2.putText(frame, f'Calib: {"ON" if calibrating else "OFF"} (K) Save(S) Load(L)', (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2); y += 20
    if fused_pos is not None:
        cv2.putText(frame, f'Fused Pos: {fused_pos[0]:.2f},{fused_pos[1]:.2f},{fused_pos[2]:.2f}', (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2); y += 20
    if ang_err is not None and pos_err is not None:
        cv2.putText(frame, f'ERR ang:{ang_err:4.1f} deg  pos:{pos_err*1000:4.0f} mm', (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2); y += 20
    cv2.putText(frame, f'Used IDs: {used_ids}   avgDM:{avg_dm:.2f}', (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2); y += 20
    cv2.putText(frame, f'Reachable: {reachable}', (10,y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
    if pair_max is not None:
        (i,j), a,p = pair_max
        cv2.putText(frame, f'PAIR max (ID{i},{j})  ang:{a:4.1f} deg  pos:{p*1000:4.0f} mm', (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,255,150), 2); y += 20
    if cycle_max is not None:
        (i,j,k), a,p = cycle_max
        cv2.putText(frame, f'CYCLE max ({i},{j},{k})  ang:{a:4.1f} deg  pos:{p*1000:4.0f} mm', (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,200,255), 2); y += 20

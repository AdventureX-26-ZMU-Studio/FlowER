"""Random A1Z motions within safe joint limits.

Usage:
  python random4look.py <次数> <间隔秒>
  python random4look.py 10 3    # 10 moves, 3s interval
"""
import sys, time, random, json, os
sys.path.insert(0, "/home/asus/zmu")

from dimos.manipulation.planning.examples.manipulation_client import _client, plan, execute
import numpy as np

# Safe zones (within URDF limits, avoiding self-collision)
LIMITS = [
    (-1.5, 1.5),      # j1
    (0.1, 2.5),       # j2
    (-2.5, 0.0),      # j3
    (-1.0, 1.0),      # j4
    (-1.0, 1.0),      # j5
    (-1.5, 1.5),      # j6
]

_FK = None
_XYZ_BOX = None

def _get_fk():
    global _FK
    if _FK is None:
        from dimos.manipulation.planning.kinematics.pinocchio_ik import PinocchioIK
        from dimos.robot.manipulators.a1z.config import A1Z_FK_MODEL, A1Z_DOF
        _FK = PinocchioIK.from_model_path(A1Z_FK_MODEL, A1Z_DOF)
    return _FK

def _joints_to_xyz(joints):
    pk = _get_fk().forward_kinematics(np.array(joints, dtype=np.float64))
    t = pk.translation
    return float(t[0]), float(t[1]), float(t[2])

def _init_box():
    global _XYZ_BOX
    path = os.path.expanduser("~/xyz_box.json")
    if os.path.exists(path):
        b = json.load(open(path))
        _XYZ_BOX = ((b["xmin"],b["ymin"],b["zmin"]), (b["xmax"],b["ymax"],b["zmax"]))

def _in_box(joints):
    if _XYZ_BOX is None: _init_box()
    if _XYZ_BOX is None: return True
    x,y,z = _joints_to_xyz(joints)
    (xmin,ymin,zmin),(xmax,ymax,zmax) = _XYZ_BOX
    return xmin<=x<=xmax and ymin<=y<=ymax and zmin<=z<=zmax

def random_joints_safe():
    for _ in range(50):
        j = random_joints()
        if _in_box(j): return j
    return random_joints()
def random_joints():
    return [random.uniform(lo, hi) for (lo, hi) in LIMITS]

def main(count: int, interval_spec: str, return_mode: int):
    # Parse: "5-9" or "[5,9]" or "5"
    s = interval_spec.replace("[","").replace("]","").replace(",","-")
    if "-" in s:
        lo, hi = map(float, s.split("-"))
        mode = f"random {lo:.1f}-{hi:.1f}s"
    lo = max(lo, 5.0); hi = max(hi, 5.0)  # min 5s
    else:
        lo = hi = float(s)
        mode = f"fixed {lo:.1f}s"
    lo = hi = max(lo, 5.0)  # min 5s
    
    print(f"Random look: {count} moves, {mode}, return={'home' if return_mode else 'start'}")
    
    # Save start position
    from dimos.manipulation.planning.examples.manipulation_client import joints
    start = joints()
    
    for i in range(count):
        target = random_joints_safe()
        tstr = [f"{x:.2f}" for x in target]
        print(f"[{i+1}/{count}] → {tstr}")

        _client.reset()
        time.sleep(0.3)
        if plan(target):
            execute()
        else:
            print(f"  plan failed, retrying...")
            _client.reset()
            time.sleep(0.3)
            if plan(target):
                execute()

        # Wait: random within range or fixed
        wait = random.uniform(lo, hi)
        time.sleep(wait)

    # Return
    if return_mode == 1:
        print("Going home...")
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        print("Returning to start...")
        target = list(start) if start else [0.0]*6
    
    _client.reset()
    time.sleep(0.5)
    plan(target)
    execute()
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No args: one random move, then hold forever
        target = random_joints_safe()
        print(f"Moving to: {[round(x,2) for x in target]}")
        _client.reset(); time.sleep(0.3)
        if plan(target): execute()
        else: _client.reset(); time.sleep(0.3); plan(target); execute()
        print("Holding...")
    elif len(sys.argv) == 2 and sys.argv[1] == "0":
        # Single "0" = go home
        _client.reset(); time.sleep(0.5)
        plan([0.0]*6); execute()
        print("Home")
    elif len(sys.argv) != 4:
        print("Usage: python random4look.py <次数> <间隔秒|min-max> <0=回起点|1=回零>")
        print("       python random4look.py              # 无限随机 hold")
        print("       python random4look.py 0            # 回正")
        sys.exit(1)
    else:
        main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))

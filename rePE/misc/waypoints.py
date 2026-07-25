"""Move to XYZ waypoints. Ctrl+C → go home and hold."""
import sys, time, signal
sys.path.insert(0, "/home/asus/zmu")
from dimos.manipulation.planning.examples.manipulation_client import _client, ik_pose, plan, execute, joints

WAYPOINTS = [
    (0.407, -0.183, 0.520),
    (0.441, -0.120, -0.267),
    (0.166, 0.412, -0.273),
    (0.177, 0.432, 0.567),
    (-0.210, 0.396, 0.588),
]

_stop = False

def on_exit(sig, frame):
    global _stop
    _stop = True
    print("\nCtrl+C: going home...")
    cur = joints()
    if cur:
        home = [0.0]*6
        for s in range(1, 6):
            t = s/5.0
            mid = [round(cur[i]+(home[i]-cur[i])*t,3) for i in range(6)]
            _client.reset(); time.sleep(0.3)
            plan(mid); execute()
            time.sleep(0.8)
    print("Home. Holding...")
    sys.exit(0)

signal.signal(signal.SIGINT, on_exit)

for i, (x, y, z) in enumerate(WAYPOINTS):
    if _stop: break
    print(f"[{i+1}/{len(WAYPOINTS)}] ({x:.3f}, {y:.3f}, {z:.3f})")

    # Try multiple approaches
    _client.reset(); time.sleep(0.5)
    r = ik_pose(x, y, z, check_collision=False)

    if r is not None and r.joint_state is not None:
        j = list(r.joint_state.position)
    else:
        print(f"  unreachable (status={r.status if r else '?'})")
        continue

    _client.reset(); time.sleep(0.3)
    if plan(j):
        execute()
        time.sleep(3)
    else:
        print(f"  plan rejected")

print("All waypoints done. Holding...")
while True:
    time.sleep(1)

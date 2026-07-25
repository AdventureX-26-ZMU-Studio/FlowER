"""Face tracking — plan/execute, traj auto-holds after completion."""
import sys, time, json, urllib.request
sys.path.insert(0, "/home/asus/zmu")
from dimos.manipulation.planning.examples.manipulation_client import joints, plan, execute

PERCEPTION = "http://127.0.0.1:8892"
STEP = 0.06
DEADZONE = 0.06

def get_face():
    try:
        resp = urllib.request.urlopen(f"{PERCEPTION}/faces", timeout=1)
        faces = json.loads(resp.read())
        if faces:
            b = faces[0]["bbox_xyxy"]
            return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    except: pass
    return None

def main():
    print("Face tracking — ctrl+c to stop")
    while True:
        cur = joints()
        if cur is None: time.sleep(0.1); continue

        face = get_face()
        if face is None: time.sleep(0.1); continue

        u, v = face
        du = u - 0.5; dv = v - 0.5
        if abs(du) < DEADZONE and abs(dv) < DEADZONE: time.sleep(0.1); continue

        target = list(cur)
        target[0] = max(-2.0, min(2.0, target[0] - du * STEP))
        target[1] = max(0.05, min(2.5, target[1] - dv * STEP * 0.7))
        target[2] = max(-2.5, min(0, target[2] + dv * STEP * 0.3))

        if plan(target):
            execute()
        time.sleep(0.1)

if __name__ == "__main__":
    main()

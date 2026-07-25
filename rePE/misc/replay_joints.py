"""Replay joint waypoints — smooth single-step per target. Ctrl+C → home."""
import sys, time, signal, json, os
sys.path.insert(0, "/home/asus/zmu")
from dimos.manipulation.planning.examples.manipulation_client import _client, plan, execute, joints

if len(sys.argv) > 1:
    FILE = sys.argv[1]
else:
    FILE = os.path.expanduser("~/waypoints.json")

if not os.path.exists(FILE):
    print("No", FILE); sys.exit(1)

waypoints = json.load(open(FILE))
print("%d waypoints loaded from %s" % (len(waypoints), FILE))

def go_home():
    cur = joints()
    if cur:
        _client.reset(); time.sleep(0.5)
        plan([0.0]*6); execute()
        print("Home.")
    sys.exit(0)

signal.signal(signal.SIGINT, lambda *_: go_home())

for i, wp in enumerate(waypoints):
    print("[%d/%d] %s" % (i+1, len(waypoints), [round(x,2) for x in wp]))
    _client.reset(); time.sleep(0.5)
    if plan(wp):
        execute()
        time.sleep(3)
    else:
        print("  plan failed")

go_home()

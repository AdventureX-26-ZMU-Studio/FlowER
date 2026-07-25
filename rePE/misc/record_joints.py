"""Append current joint position to waypoints. Ctrl+C to save."""
import sys, time, signal, json, os
sys.path.insert(0, "/home/asus/zmu")
from dimos.manipulation.planning.examples.manipulation_client import joints

FILE = os.path.expanduser("~/waypoints.json")

def save():
    j = joints()
    if not j: return
    data = []
    if os.path.exists(FILE):
        data = json.load(open(FILE))
    data.append([round(x, 4) for x in j])
    json.dump(data, open(FILE, "w"), indent=2)
    print("+1 → %d total" % len(data))

signal.signal(signal.SIGINT, lambda *_: (save(), sys.exit(0)))
print("Ctrl+C to record. Run replay to play back.")
while True:
    time.sleep(1)

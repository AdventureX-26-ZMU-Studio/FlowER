import subprocess, os, glob, json, time, threading, signal
from flask import render_template, request, jsonify
from app.blueprints.control import bp

VENV = "/home/asus/dimos/.venv/bin/python"
ZMU = "/home/asus/zmu"
SCRIPT_RANDOM = "/home/asus/zmu/rePE/misc/random4look.py"
SCRIPT_REPLAY = "/home/asus/zmu/rePE/misc/replay_joints.py"
RECORDS_DIR = os.path.expanduser("~/records")

os.makedirs(RECORDS_DIR, exist_ok=True)

# Active recording state
_recording = {"active": False, "name": "", "joints": [], "thread": None}

def _record_loop():
    sys.path.insert(0, ZMU)
    from dimos.manipulation.planning.examples.manipulation_client import joints
    while _recording["active"]:
        j = joints()
        if j: _recording["joints"].append([round(x, 4) for x in j])
        time.sleep(0.5)

def _run_async(cmd):
    env = os.environ.copy(); env["PYTHONPATH"] = ZMU
    try:
        subprocess.Popen(cmd, env=env, cwd=ZMU, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify({"ok": True, "cmd": " ".join(cmd)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

def _run_sync(cmd):
    env = os.environ.copy(); env["PYTHONPATH"] = ZMU
    try:
        r = subprocess.run(cmd, env=env, cwd=ZMU, capture_output=True, text=True, timeout=30)
        return jsonify({"ok": r.returncode == 0, "cmd": " ".join(cmd),
                        "stdout": r.stdout[-200:], "stderr": r.stderr[-200:]})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": True, "cmd": " ".join(cmd), "stdout": "running..."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@bp.route("/")
def panel():
    return render_template("control/panel.html")

@bp.route("/random", methods=["POST"])
def random_move():
    count = request.form.get("count", "1")
    interval = request.form.get("interval", "3")
    return_home = request.form.get("return_home", "1")
    cmd = [VENV, SCRIPT_RANDOM]
    if count == "0": cmd.append("0")
    elif count == "" or count is None: pass
    else: cmd.extend([count, interval, return_home])
    return _run_sync(cmd)

@bp.route("/replay8", methods=["POST"])
def replay_8():
    return _run_async([VENV, SCRIPT_REPLAY, os.path.expanduser("~/waypoints_8cube.json")])

@bp.route("/replay3", methods=["POST"])
def replay_3():
    return _run_async([VENV, SCRIPT_REPLAY, os.path.expanduser("~/waypoints.json")])

# ── Record API ──
@bp.route("/records", methods=["GET"])
def list_records():
    files = sorted(glob.glob(f"{RECORDS_DIR}/*.json"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            with open(f) as fp: data = json.load(fp)
            count = len(data) if isinstance(data, list) else 0
        except: count = 0
        result.append({"name": name, "count": count, "mtime": os.path.getmtime(f)})
    return jsonify(result)

@bp.route("/records/start", methods=["POST"])

@bp.route("/records/capture", methods=["POST"])
def capture_point():
    import sys; sys.path.insert(0, ZMU)
    from dimos.manipulation.planning.examples.manipulation_client import joints
    j = joints()
    if not j: return jsonify({"ok": False, "error": "No joint data"})
    return jsonify({"ok": True, "joints": [round(x, 4) for x in j]})

@bp.route("/records/save", methods=["POST"])
def save_record():
    name = request.form.get("name", "").strip()
    joints_str = request.form.get("joints", "[]")
    try:
        joints = json.loads(joints_str)
    except:
        return jsonify({"ok": False, "error": "Invalid joints data"})
    if not name: return jsonify({"ok": False, "error": "Name required"})
    if len(joints) == 0: return jsonify({"ok": False, "error": "No points"})
    path = os.path.join(RECORDS_DIR, f"{name}.json")
    json.dump(joints, open(path, "w"), indent=2)
    return jsonify({"ok": True, "name": name, "points": len(joints)})

def start_recording():
    if _recording["active"]:
        return jsonify({"ok": False, "error": "Already recording"})
    name = request.form.get("name", "").strip()
    if not name: return jsonify({"ok": False, "error": "Name required"})
    # Validate illegal characters
    import re as _re
    if _re.search(r"[<>:/|?*&0@!`]", name):
        return jsonify({"ok": False, "error": "Name contains illegal characters"})
    # Check duplicate
    path = os.path.join(RECORDS_DIR, f"{name}.json")
    if os.path.exists(path):
        return jsonify({"ok": False, "error": f"Record {name} already exists"})
    _recording["active"] = True
    _recording["name"] = name
    _recording["joints"] = []
    import sys
    _recording["thread"] = threading.Thread(target=_record_loop, daemon=True)
    _recording["thread"].start()
    return jsonify({"ok": True, "name": name})
@bp.route("/records/stop", methods=["POST"])
def stop_recording():
    if not _recording["active"]:
        return jsonify({"ok": False, "error": "Not recording"})
    _recording["active"] = False
    name = _recording["name"]
    joints = _recording["joints"]
    path = os.path.join(RECORDS_DIR, f"{name}.json")
    json.dump(joints, open(path, "w"), indent=2)
    _recording["joints"] = []
    return jsonify({"ok": True, "name": name, "points": len(joints), "path": path})

@bp.route("/records/status", methods=["GET"])
def recording_status():
    return jsonify({"active": _recording["active"], "name": _recording["name"],
                    "points": len(_recording["joints"]) if _recording["active"] else 0})

@bp.route("/records/replay", methods=["POST"])
def replay_record():
    name = request.form.get("name", "").strip()
    if not name: return jsonify({"ok": False, "error": "Name required"})
    path = os.path.join(RECORDS_DIR, f"{name}.json")
    if not os.path.exists(path): return jsonify({"ok": False, "error": "Not found"})
    return _run_async([VENV, SCRIPT_REPLAY, path])

@bp.route("/gripper", methods=["POST"])
def gripper_control():
    action = request.form.get("action", "open")
    import sys; sys.path.insert(0, ZMU)
    from dimos.manipulation.planning.examples.manipulation_client import gripper
    if action == "close":
        result = gripper(0.0)
    elif action == "open":
        result = gripper(0.08)
    elif action == "grasp":
        gripper(0.0); import time; time.sleep(0.5); gripper(0.08)
        result = "Grasp cycle done"
    else:
        return jsonify({"ok": False, "error": "Unknown action"})
    return jsonify({"ok": True, "result": str(result)})

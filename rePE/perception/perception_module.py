"""rePE perception — FaceMesh on :8892. VisionHub WS → MediaPipe → MJPEG + JSON."""
import json, time, threading, base64, struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    pass
from dataclasses import dataclass
import cv2, numpy as np
import mediapipe as mp

# ── config ──
@dataclass
class Config:
    vision_host: str = "127.0.0.1"
    vision_port: int = 8890
    http_host: str = "0.0.0.0"
    http_port: int = 8892
    detect_every: int = 3
    face_min_conf: float = 0.6
    max_faces: int = 4

cfg = Config()

# ── state ──
JPEG = None
FACES: list[dict] = []
FPS_LOG = [time.time()]
FM_MS = 0
WS_ALIVE = True

# ── model ──
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=cfg.max_faces,
    refine_landmarks=False, min_detection_confidence=cfg.face_min_conf)

hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ── WebSocket ──
def ws_loop():
    global JPEG, WS_ALIVE
    while True:
        try:
            sock = __import__("socket").socket(); sock.settimeout(5)
            sock.connect((cfg.vision_host, cfg.vision_port))
            key = base64.b64encode(b"1234567890123456").decode()
            req = f"GET /v1/streams/ws?service_name=perception&streams=rgb HTTP/1.1\r\nHost: {cfg.vision_host}:{cfg.vision_port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            sock.send(req.encode())
            if b"101" not in sock.recv(4096): sock.close(); time.sleep(0.5); continue
            buf = b""; WS_ALIVE = True
            while True:
                try:
                    data = sock.recv(8192)
                    if not data: break
                    buf += data
                    while len(buf) >= 2:
                        opcode = buf[0] & 0x0F
                        if opcode == 8: raise Exception("close")
                        if opcode == 9: sock.send(bytearray([0x8A,0x00])); buf=buf[2:]; continue
                        mask = buf[1] & 0x80; length = buf[1] & 0x7F; off = 2
                        if length==126: length=struct.unpack(">H",buf[2:4])[0]; off=4
                        elif length==127: length=struct.unpack(">Q",buf[2:10])[0]; off=10
                        if mask: mk=buf[off:off+4]; off+=4
                        if len(buf) < off+length: break
                        payload=buf[off:off+length]
                        if mask: payload=bytes(b^mk[i%4] for i,b in enumerate(payload))
                        buf=buf[off+length:]
                        if opcode==1:
                            try:
                                d=json.loads(payload.decode())
                                b64=d.get("rgb_jpeg_b64")
                                if b64:
                                    JPEG=base64.b64decode(b64)
                                    FPS_LOG.append(time.time())
                                    if len(FPS_LOG)>60: FPS_LOG.pop(0)
                            except: pass
                except: break
            sock.close(); WS_ALIVE = False
        except: time.sleep(0.3)

# ── detection ──
_frame_n = 0
HANDS: list[dict] = []

def detect(img_bgr):
    global FACES, HANDS, _frame_n, FM_MS
    _frame_n += 1
    if _frame_n % cfg.detect_every != 0:
        return
    t0 = time.time()
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = face_mesh.process(rgb)
        hand_result = hands_detector.process(rgb)
        rgb.flags.writeable = True
        faces = []
        if result.multi_face_landmarks:
            for i, lm in enumerate(result.multi_face_landmarks):
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                faces.append({
                    "id": f"face_{i}", "present": True,
                    "bbox_xyxy": [round(min(xs),3), round(min(ys),3),
                                  round(max(xs),3), round(max(ys),3)],
                    "confidence": 1.0, "depth_m": None,
                })
        FACES = faces
        hands = []
        if hand_result.multi_hand_landmarks:
            handedness_list = hand_result.multi_handedness or []
            for i, lm in enumerate(hand_result.multi_hand_landmarks):
                hlabel = "Unknown"
                if i < len(handedness_list) and handedness_list[i].classification:
                    hlabel = handedness_list[i].classification[0].label
                wrist = lm.landmark[0]
                hands.append({
                    "id": f"hand_{i}", "handedness": hlabel,
                    "x": round(wrist.x, 3), "y": round(wrist.y, 3),
                    "confidence": 1.0,
                })
        HANDS = hands
    except: pass
    FM_MS = int((time.time()-t0)*1000)

def draw(img_bgr):
    if not FACES and not HANDS:
        return
    h, w = img_bgr.shape[:2]
    try:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        if FACES:
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                for lm in result.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img_bgr, lm, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        if HANDS:
            hand_result = hands_detector.process(rgb)
            if hand_result.multi_hand_landmarks:
                for lm in hand_result.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img_bgr, lm, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style())
        rgb.flags.writeable = True
    except: pass

# ── HTTP ──
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)
        res = q.get("res",["640x360"])[0]; fps = q.get("fps",["30"])[0]
        tw,th = (int(x) for x in res.split("x"))
        tfps = int(fps); interval = 1.0/tfps if tfps>0 else 0

        if self.path == "/faces":
            self._json(FACES)
        elif self.path == "/hands":
            self._json(HANDS)
        elif self.path == "/health":
            self._json({"status":"ok","faces":len(FACES),"hands":len(HANDS),"fm_ms":FM_MS,"ws":WS_ALIVE})
        elif self.path.startswith("/s") or self.path.startswith("/fm"):
            do_fm = self.path.startswith("/fm"); last = 0
            try:
                self.send_response(200)
                self.send_header("Content-Type","multipart/x-mixed-replace; boundary=f")
                self.send_header("Cache-Control","no-cache")
                self.end_headers()
                while True:
                    try:
                        if JPEG and time.time()-last >= interval:
                            img = cv2.imdecode(np.frombuffer(JPEG,np.uint8),cv2.IMREAD_COLOR)
                            if img is None: time.sleep(0.01); continue
                            if img.shape[1]!=tw or img.shape[0]!=th:
                                img = cv2.resize(img,(tw,th),interpolation=cv2.INTER_NEAREST)
                            detect(img)
                            if do_fm: draw(img)
                            _,jpg=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,55])
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n"+jpg.tobytes()+b"\r\n")
                            self.wfile.flush(); last=time.time()
                        time.sleep(0.005)
                    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError,OSError): break
            except: pass
        else:
            dt=FPS_LOG[-1]-FPS_LOG[0] if len(FPS_LOG)>1 else 1
            fps_val=(len(FPS_LOG)-1)/dt if dt>0 else 0
            ws_warn='<b style=color:red>⚠ WS</b>' if not WS_ALIVE else ''
            self.send_response(200); self.send_header("Content-Type","text/html"); self.end_headers()
            self.wfile.write(f"""<!doctype html><html><head><meta charset=utf-8><title>Perception :8892</title>
<style>body{{font-family:system-ui;background:#0d1117;color:#c9d1d9;padding:20px;text-align:center}}
h1{{color:#58a6ff}}img{{max-width:100%;border-radius:8px}}
a{{font-size:14px;margin:4px;color:#58a6ff}}p,b{{font-size:13px;color:#8b949e}}</style></head>
<body><h1>Perception :8892</h1><p>FPS:{fps_val:.1f} | FM:{FM_MS}ms | Faces:{len(FACES)} {ws_warn}</p>
<a href=/s>Raw</a> | <a href=/fm>FaceMesh</a>
<form oninput="u()">Res:<select id=r onchange="u()"><option>{res}</option><option>320x180</option><option>640x360</option></select>
FPS:<select id=f onchange="u()"><option>{fps}</option><option>5</option><option>10</option><option>15</option><option>30</option></select></form>
<img src={self.path} style=max-width:100% onerror="this.src=this.src">
<script>function u(){{location.href=location.pathname+"?res="+document.getElementById("r").value+"&fps="+document.getElementById("f").value}};</script></body></html>""".encode())

    def _json(self,data):
        b=json.dumps(data,default=str).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)

    def log_message(self,*a): pass

# ── main ──
threading.Thread(target=ws_loop,daemon=True).start()

# dimos Module stub (blueprint-ready)
try:
    from dimos.core.module import Module, ModuleConfig
    from dimos.core.core import rpc
except ImportError:
    class Module:
        config = None
        def start(self): pass
        def stop(self): pass
    class ModuleConfig:
        pass
    def rpc(fn):
        fn.__rpc__ = True
        return fn

@dataclass
class PerceptionModuleConfig(ModuleConfig):
    http_port: int = cfg.http_port

class PerceptionModule(Module):
    config: PerceptionModuleConfig
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
    @rpc
    def start(self):
        threading.Thread(target=lambda: ThreadingHTTPServer((cfg.http_host,cfg.http_port),H).serve_forever(),daemon=True).start()
    @rpc
    def stop(self): pass

if __name__=="__main__":
    print(f"Perception :{cfg.http_port}")
    ThreadingHTTPServer((cfg.http_host,cfg.http_port),H).serve_forever()

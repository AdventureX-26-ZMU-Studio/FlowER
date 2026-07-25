"""tracking — dimos Module for face following via eef_twist.

Blueprint autoconnect wires: tracking.ee_twist → coordinator.coordinator_ee_twist_command
"""
from __future__ import annotations

import threading, time, json, urllib.request

try:
    from dimos.core.module import Module, ModuleConfig
    from dimos.core.core import rpc
    from dimos.core.stream import Out
    from dimos.msgs.geometry_msgs.TwistStamped import TwistStamped
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
except ImportError:
    # Standalone test stub
    class Module: pass
    class ModuleConfig: pass
    def rpc(fn): return fn
    class Out: pass
    TwistStamped = Twist = Vector3 = None

PERCEPTION = "http://127.0.0.1:8892"
GAIN = 2.0
DEADZONE = 0.05
INTERVAL = 0.05


class TrackingConfig(ModuleConfig):
    perception_url: str = PERCEPTION
    gain: float = GAIN
    deadzone: float = DEADZONE
    interval: float = INTERVAL

class TrackingModule(Module):
    config: TrackingConfig
    coordinator_ee_twist_command: Out[TwistStamped]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._running = False
        self._thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while self._running:
            u, v = self._get_face()
            if u is None:
                self._publish(0, 0)
                time.sleep(self.config.interval)
                continue

            du = u - 0.5
            dv = v - 0.5

            if abs(du) < self.config.deadzone and abs(dv) < self.config.deadzone:
                self._publish(0, 0)
                time.sleep(self.config.interval)
                continue

            vy = -du * self.config.gain
            vz = -dv * self.config.gain
            self._publish(vy, vz)
            time.sleep(self.config.interval)

    def _get_face(self):
        try:
            resp = urllib.request.urlopen(f"{self.config.perception_url}/faces", timeout=1)
            faces = json.loads(resp.read())
            if faces:
                b = faces[0]["bbox_xyxy"]
                return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        except Exception:
            pass
        return None, None

    def _publish(self, vy: float, vz: float) -> None:
        if TwistStamped is None:
            return
        self.coordinator_ee_twist_command.publish(TwistStamped(
            twist=Twist(linear=Vector3(0, vy, vz), angular=Vector3(0, 0, 0)),
            frame_id="eef_twist_arm"))

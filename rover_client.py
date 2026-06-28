"""Laptop-side client to talk to the Pi body server. Also a MOCK mode for dev
without the robot (uses webcam + prints commands), so you can build the brain
before the rover is wired up.

    USE_MOCK=1 ./.venv/bin/python main.py "red cup"   # no robot needed
"""
import os
import time

import requests

import config

USE_MOCK = os.environ.get("USE_MOCK") == "1"

_base = f"http://{config.ROVER_HOST}:{config.ROVER_PORT}"


# ---------- REAL rover ----------
def _real_get_frame() -> bytes:
    r = requests.get(f"{_base}/frame", timeout=2)
    r.raise_for_status()
    return r.content


def _real_send_cmd(cmd: dict):
    requests.post(f"{_base}/cmd", json=cmd, timeout=2)


# ---------- MOCK rover (laptop webcam, no motors) ----------
_mock_cap = None


def _mock_get_frame() -> bytes:
    import cv2
    global _mock_cap
    if _mock_cap is None:
        _mock_cap = cv2.VideoCapture(0)
    ok, frame = _mock_cap.read()
    if not ok:
        raise RuntimeError("mock webcam read failed")
    ok2, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _mock_send_cmd(cmd: dict):
    print(f"  [MOCK MOTOR] {cmd}")


# ---------- public ----------
get_frame = _mock_get_frame if USE_MOCK else _real_get_frame
send_cmd = _mock_send_cmd if USE_MOCK else _real_send_cmd


def do_action(action: str):
    """Map a high-level action -> motor command, run it for MOVE_PULSE_SEC, then stop."""
    cmd_fn = config.ACTION_TO_CMD.get(action, config.cmd_stop)
    send_cmd(cmd_fn())
    if action not in ("stop", "done"):
        time.sleep(config.MOVE_PULSE_SEC)
        send_cmd(config.cmd_stop())

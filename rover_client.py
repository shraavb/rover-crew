"""Laptop-side client to drive the rover body. Three backends:

    real  (default)  HTTP to the Pi body server -> JSON over serial to the ESP32
    mock  USE_MOCK=1 laptop webcam + prints motor commands (no robot)
    sim   USE_SIM=1  Cyberwave digital twin (no physical rover; runs headless)

    USE_MOCK=1 ./.venv/bin/python main.py "red cup"   # no robot needed
    USE_SIM=1  ./.venv/bin/python main.py "red cup"   # Cyberwave digital twin
"""
import os
import time

import requests

import config

USE_MOCK = os.environ.get("USE_MOCK") == "1"
USE_SIM = os.environ.get("USE_SIM") == "1"
MODE = "sim" if USE_SIM else "mock" if USE_MOCK else "real"

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


# ---------- SIM rover (Cyberwave digital twin) ----------
_twin = None


def _sim_twin():
    global _twin
    if _twin is None:
        from cyberwave import Cyberwave  # lazy: only needed in sim mode
        cw = Cyberwave()                  # CYBERWAVE_API_KEY from env
        cw.affect("simulation")
        _twin = cw.twin(config.CW_TWIN)
    return _twin


def _sim_get_frame() -> bytes:
    import cv2
    frame = _sim_twin().capture_frame("numpy")  # BGR ndarray
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("sim frame encode failed")
    return buf.tobytes()


def _sim_send_cmd(cmd: dict):
    # Real rover speaks differential L/R; the twin is high-level. The only dict
    # cmd that reaches here is the emergency stop from main()'s finally block.
    _sim_twin().move_forward(distance=0.0)


def _sim_do_action(action: str):
    twin = _sim_twin()
    if action == "forward":
        twin.move_forward(distance=config.SIM_STEP_M)
    elif action == "turn_left":
        twin.turn_left(angle=config.SIM_TURN_RAD)
    elif action == "turn_right":
        twin.turn_right(angle=config.SIM_TURN_RAD)
    else:  # stop, done, back, anything unknown -> halt
        twin.move_forward(distance=0.0)


# ---------- public ----------
get_frame = {"sim": _sim_get_frame, "mock": _mock_get_frame, "real": _real_get_frame}[MODE]
send_cmd = {"sim": _sim_send_cmd, "mock": _mock_send_cmd, "real": _real_send_cmd}[MODE]


def do_action(action: str):
    """Map a high-level action -> motor command, run it for MOVE_PULSE_SEC, then stop."""
    if MODE == "sim":
        # Twin motion is a discrete displacement (metres/radians), no pulse needed.
        _sim_do_action(action)
        return
    cmd_fn = config.ACTION_TO_CMD.get(action, config.cmd_stop)
    send_cmd(cmd_fn())
    if action not in ("stop", "done"):
        time.sleep(config.MOVE_PULSE_SEC)
        send_cmd(config.cmd_stop())

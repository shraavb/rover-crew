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
        if config.CW_TWIN_ID:
            _twin = cw.twin(twin_id=config.CW_TWIN_ID)
        else:
            _twin = cw.twin(config.CW_TWIN, environment_id=config.CW_ENV)
    return _twin


def _sim_get_frame() -> bytes:
    # In simulation mode frames come from the cloud render (source="cloud" is the
    # get_frame default); "bytes" returns JPEG directly.
    jpg = _sim_twin().get_frame("bytes", mock=config.SIM_MOCK_FRAME)
    # The endpoint fail-softs to a JSON error body (not an image) when the sim has
    # no rendered frame. Reject non-JPEG so we never ship junk to the VLM.
    if not jpg or jpg[:3].hex() != "ffd8ff":
        detail = jpg[:200].decode("utf-8", "replace") if jpg else "empty response"
        raise RuntimeError(
            "sim returned no camera frame. Start the simulation in the dashboard "
            "(SIMULATE tab) so the camera renders, or set SIM_MOCK_FRAME=1 to test "
            f"the loop with a placeholder. server: {detail}"
        )
    return jpg


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
_GET_FRAME = {"sim": _sim_get_frame, "mock": _mock_get_frame, "real": _real_get_frame}
_SEND_CMD = {"sim": _sim_send_cmd, "mock": _mock_send_cmd, "real": _real_send_cmd}

# Hybrid: the Cyberwave sim has no synthetic camera, so SIM_FRAME=webcam takes
# perception from the laptop webcam while motion still drives the twin.
if MODE == "sim" and config.SIM_FRAME == "webcam":
    get_frame = _mock_get_frame
else:
    get_frame = _GET_FRAME[MODE]
send_cmd = _SEND_CMD[MODE]


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

"""Central config. Edit ROVER_HOST + SERIAL_PORT + command map for your Waveshare UGV.

Loads a .env file (if present) so CEREBRAS_API_KEY / CYBERWAVE_API_KEY can live
there instead of being exported each shell. Imported first by every module, so
keys are in os.environ before agents.py / rover_client.py read them.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from cwd up the tree; real env vars still win
except ImportError:
    pass  # dotenv optional; fall back to exported env vars

# ---- Networking ----
# IP of the rover's Raspberry Pi on your WiFi (run `hostname -I` on the Pi).
ROVER_HOST = "10.0.0.194"   # Waveshare UGV Beast Pi (ugvrpi.local) on home wifi
ROVER_PORT = 5000

# ---- Model ----
MODEL = "gemma-4-31b"

# ---- Cyberwave sim backend (USE_SIM=1) ----
# Digital twin of the UGV Beast. Auth via CYBERWAVE_API_KEY env var; create the
# twin in the dashboard (Add from Catalog -> UGV Beast). Motion is high-level
# (metres / radians), not the differential L/R serial cmds the real rover uses.
# Preferred: connect to an exact twin by UUID (from the dashboard, twin -> Copy
# UUID). Set CYBERWAVE_TWIN_ID in .env. Bypasses env/asset resolution entirely.
CW_TWIN_ID = os.environ.get("CYBERWAVE_TWIN_ID")

# Fallback when no twin UUID: look up by asset in a given environment.
CW_TWIN = "waveshare/ugv-beast"
# Must be a UUID or a FULL unified slug "<workspace>/<env>" (a bare env name has
# no '/', so the SDK won't resolve it). Env var wins.
CW_ENV = os.environ.get("CYBERWAVE_ENVIRONMENT_ID") or "shraavasti-bhats-workspace/ugv-rover"
SIM_STEP_M = 0.3      # metres per forward step   (SDK caps at 1.0)
SIM_TURN_RAD = 0.35   # radians per turn step (small enough to avoid overshoot
                      # oscillation when centering on an off-axis target)
# SIM_MOCK_FRAME=1 -> use the SDK's deterministic placeholder frame instead of a
# real render. Validates the loop plumbing when the sim isn't rendering yet.
SIM_MOCK_FRAME = os.environ.get("SIM_MOCK_FRAME") == "1"
# The Cyberwave sim simulates MOTION only, not a camera. So sim perception needs
# a real camera. SIM_FRAME=webcam -> hybrid: laptop webcam for frames, motion
# still driven to the twin. "cloud" (default) expects a real camera streamed to
# the twin via the edge SDK.
SIM_FRAME = (os.environ.get("SIM_FRAME") or "cloud").lower()

# ---- Rate limit ----
# Cerebras enforces ~100 requests/min. Each control loop makes 2 API calls
# (perceive + plan; safety is local). Blowing past the limit returns HTTP 429,
# which is why a naive fixed sleep "works" at 1200ms but stalls at 600ms.
# The limiter (ratelimit.py) throttles to MAX_RPM and backs off on 429 so the
# loop degrades smoothly instead of erroring. Keep margin under the hard 100.
MAX_RPM = float(os.environ.get("MAX_RPM") or 90.0)  # raise for the hackathon
                                                    # rate increase to show full
                                                    # Cerebras speed; 429s back off

# ---- Control loop ----
LOOP_HZ = 3.0                 # target perceive->act cycles per second
                              # NOTE: actual rate is capped by MAX_RPM via the
                              # limiter. 2 calls/loop * LOOP_HZ must stay under
                              # MAX_RPM/60 or the limiter throttles the loop.
MOVE_PULSE_SEC = 0.6          # how long each motion command runs before re-looking
TURN_SPEED = 0.25             # wheel speed for turning (m/s-ish, tune)
FWD_SPEED = 0.25              # wheel speed forward

# ---- Stop-look-move timing (shared by every behavior in behaviors.py) ----
# Each step pulses one motion for MOVE_TIME, stops, then waits SETTLE_TIME so the
# camera frame used for the NEXT decision is sharp (continuous driving blurred
# every frame and perception went blind). Env-overridable.
MOVE_TIME = float(os.environ.get("MOVE_TIME") or 0.35)
SETTLE_TIME = float(os.environ.get("SETTLE_TIME") or 0.35)
MAX_STEPS = int(os.environ.get("MAX_STEPS") or 60)
# Forward approach steps forced while the target is mid/far before 'done' is
# honoured (Gemma over-reports 'near', else the rover stops across the room).
MIN_APPROACH = int(os.environ.get("MIN_APPROACH") or 3)
# First search-turn direction before the target has ever been seen.
SEARCH_DIR = (os.environ.get("SEARCH_DIR") or "right").lower()
# Turn calibration: how many stop-look-move turn pulses make ~a quarter turn.
# Tune on hardware (a pulse = TURN_SPEED for MOVE_TIME). 180deg = 2x this.
TURN_PULSES_90 = int(os.environ.get("TURN_PULSES_90") or 3)
# Final close-in: Gemma calls 'near' ~20cm out; drive this many extra short
# forward pulses once near+centered to finish ~5-10cm from the target. Tune up
# for closer, down (0) to stop as soon as 'near'.
CLOSE_STEPS = int(os.environ.get("CLOSE_STEPS") or 1)

# ---- Waveshare UGV serial command map (VERIFY against your model's JSON cmd set) ----
# Waveshare UGV uses JSON over serial to the ESP32 sub-controller.
# T:1 = differential speed control, L=left wheels, R=right wheels.
def cmd_drive(left: float, right: float) -> dict:
    return {"T": 1, "L": left, "R": right}

def cmd_forward() -> dict:
    return cmd_drive(FWD_SPEED, FWD_SPEED)

def cmd_back() -> dict:
    return cmd_drive(-FWD_SPEED, -FWD_SPEED)

def cmd_turn_left() -> dict:
    return cmd_drive(-TURN_SPEED, TURN_SPEED)

def cmd_turn_right() -> dict:
    return cmd_drive(TURN_SPEED, -TURN_SPEED)

# Veer = drive forward while curving toward one side (inner wheel slowed). Used to
# center an off-center target while still advancing, instead of spinning in place
# (which overshoots and makes the rover orbit the target).
VEER_INNER = 0.45                  # inner-wheel fraction of FWD_SPEED
def cmd_veer_left() -> dict:
    return cmd_drive(FWD_SPEED * VEER_INNER, FWD_SPEED)

def cmd_veer_right() -> dict:
    return cmd_drive(FWD_SPEED, FWD_SPEED * VEER_INNER)

def cmd_stop() -> dict:
    return cmd_drive(0.0, 0.0)

ACTION_TO_CMD = {
    "forward": cmd_forward,
    "turn_left": cmd_turn_left,
    "turn_right": cmd_turn_right,
    "veer_left": cmd_veer_left,
    "veer_right": cmd_veer_right,
    "back": cmd_back,
    "stop": cmd_stop,
    "done": cmd_stop,
}

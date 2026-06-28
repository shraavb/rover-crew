"""Central config. Edit ROVER_HOST + SERIAL_PORT + command map for your Waveshare UGV."""

# ---- Networking ----
# IP of the rover's Raspberry Pi on your WiFi (run `hostname -I` on the Pi).
ROVER_HOST = "192.168.1.50"   # <-- CHANGE ME
ROVER_PORT = 5000

# ---- Model ----
MODEL = "gemma-4-31b"

# ---- Rate limit ----
# Cerebras enforces ~100 requests/min. Each control loop makes 2 API calls
# (perceive + plan; safety is local). Blowing past the limit returns HTTP 429,
# which is why a naive fixed sleep "works" at 1200ms but stalls at 600ms.
# The limiter (ratelimit.py) throttles to MAX_RPM and backs off on 429 so the
# loop degrades smoothly instead of erroring. Keep margin under the hard 100.
MAX_RPM = 90.0

# ---- Control loop ----
LOOP_HZ = 3.0                 # target perceive->act cycles per second
                              # NOTE: actual rate is capped by MAX_RPM via the
                              # limiter. 2 calls/loop * LOOP_HZ must stay under
                              # MAX_RPM/60 or the limiter throttles the loop.
MOVE_PULSE_SEC = 0.6          # how long each motion command runs before re-looking
TURN_SPEED = 0.25             # wheel speed for turning (m/s-ish, tune)
FWD_SPEED = 0.25              # wheel speed forward

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

def cmd_stop() -> dict:
    return cmd_drive(0.0, 0.0)

ACTION_TO_CMD = {
    "forward": cmd_forward,
    "turn_left": cmd_turn_left,
    "turn_right": cmd_turn_right,
    "back": cmd_back,
    "stop": cmd_stop,
    "done": cmd_stop,
}

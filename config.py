"""Central config. Edit ROVER_HOST + SERIAL_PORT + command map for your Waveshare UGV."""

# ---- Networking ----
# IP of the rover's Raspberry Pi on your WiFi (run `hostname -I` on the Pi).
ROVER_HOST = "192.168.1.50"   # <-- CHANGE ME
ROVER_PORT = 5000

# ---- Model ----
MODEL = "gemma-4-31b"

# ---- Control loop ----
LOOP_HZ = 3.0                 # target perceive->act cycles per second
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

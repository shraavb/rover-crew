"""Self-contained MuJoCo sim backend (USE_MJC=1).

Unlike the Cyberwave twin, this renders a real onboard camera, so the Gemma
crew actually *sees* the simulated world (a red target box) and can drive to it
and reach `done`. The rover is dead-reckoned (kinematic free joint) from the
crew's discrete actions. A chase-camera window (with the onboard view inset)
shows the robot moving for the demo recording.

    USE_MJC=1 ./.venv/bin/python main.py "red box"
"""
import math
import time

import cv2
import mujoco
import numpy as np

import config

# Gravity off: we set the pose kinematically via mj_forward (no dynamics), so
# the rover holds exactly the pose we author each step.
_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
    <map fogstart="8" fogend="18" haze="0.15"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.6 0.75"
             rgb2="0.85 0.88 0.92" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.32 0.34 0.36"
             rgb2="0.26 0.28 0.30" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="16 16" reflectance="0.15"/>
    <texture name="wood" type="cube" builtin="flat" rgb1="0.52 0.36 0.20"
             width="64" height="64"/>
    <material name="crate" texture="wood" specular="0.2" shininess="0.3"/>
    <material name="wall" rgba="0.6 0.62 0.66 1" specular="0.1"/>
  </asset>
  <worldbody>
    <light directional="true" pos="2 -2 6" dir="-0.3 0.3 -1" castshadow="true"
           diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="18 18 0.1" material="grid"/>

    <!-- distant perimeter walls: backdrop realism, beyond the rover bounds -->
    <geom type="box" pos="9 0 0.6" size="0.15 10 0.6" material="wall"/>
    <geom type="box" pos="-6 0 0.6" size="0.15 10 0.6" material="wall"/>
    <geom type="box" pos="1.5 9 0.6" size="10 0.15 0.6" material="wall"/>
    <geom type="box" pos="1.5 -9 0.6" size="10 0.15 0.6" material="wall"/>

    <body name="rover" pos="0 0 0.12">
      <freejoint/>
      <geom type="box" size="0.22 0.16 0.10" rgba="0.2 0.45 0.95 1"/>
      <geom type="box" size="0.05 0.17 0.02" pos="0.18 0 0.06" rgba="0.1 0.1 0.1 1"/>
      <camera name="onboard" pos="0.22 0 0.16" xyaxes="0 -1 0 0 0 1"/>
    </body>

    <!-- TARGET: a TALL ORANGE cup so its top stays visible OVER the shorter
         crates -- the rover keeps its goal anchored while avoiding obstacles. -->
    <body name="target" pos="4.8 1.2 0.5">
      <geom type="cylinder" size="0.20 0.5" rgba="1.0 0.5 0.05 1"/>
    </body>

    <!-- OBSTACLE on the route (shorter than the cup): when close ahead the SAFETY
         agent vetoes forward and steers around it; the cup stays visible above. -->
    <!-- BASKET: a tall laundry hamper (blue body, orange band) the rover goes
         around -- mirrors the real room (basketball hamper in front of the case). -->
    <body name="basket" pos="2.3 -1.0 0.40">
      <geom type="box" size="0.26 0.26 0.40" rgba="0.12 0.18 0.55 1"/>
      <geom type="box" size="0.27 0.27 0.10" pos="0 0 0.12" rgba="0.9 0.45 0.1 1"/>
    </body>
    <!-- scenery crates to the sides (realism; not on the route) -->
    <geom name="crate_a" type="box" pos="3.8 -2.2 0.20" size="0.35 0.35 0.20" material="crate"/>
    <geom name="crate_b" type="box" pos="1.0 3.0 0.20" size="0.35 0.35 0.20" material="crate"/>

    <!-- DISTRACTOR CUPS: same shape, different colours (incl. red & yellow, the
         confusable neighbours of orange) -> Gemma must pick ORANGE, not these. -->
    <body name="cup_red" pos="3.4 -1.6 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.85 0.08 0.08 1"/>
    </body>
    <body name="cup_yellow" pos="2.2 2.3 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.9 0.85 0.1 1"/>
    </body>
    <body name="cup_green" pos="4.4 -2.6 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.15 0.75 0.2 1"/>
    </body>
    <body name="cup_blue" pos="1.3 -1.8 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.15 0.3 0.9 1"/>
    </body>
    <body name="cup_purple" pos="5.6 -1.4 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.55 0.15 0.8 1"/>
    </body>

    <!-- another object type (not a cup) for extra clutter -->
    <body name="ball_white" pos="2.9 3.2 0.25">
      <geom type="sphere" size="0.25" rgba="0.9 0.9 0.92 1"/>
    </body>

    <camera name="chase" mode="trackcom" pos="2.2 -6.5 5.2"/>
  </worldbody>
</mujoco>
"""

import atexit
import os

_SHOW = os.environ.get("MJC_WINDOW", "1") == "1"  # MJC_WINDOW=0 for headless
# Smooth each discrete move into a glide so the viewer can take in the scene
# (multiple coloured cups) instead of the rover teleporting per step. Only when
# the window is shown; headless runs jump straight to the final pose (fast tests).
_ANIM_FRAMES = int(os.environ.get("SIM_ANIM_FRAMES") or 8)
_ANIM_DT = float(os.environ.get("SIM_ANIM_DT") or 0.035)  # seconds per tween frame
# glide per move ~= FRAMES*DT (~0.28s). Lower SIM_ANIM_DT / FRAMES for faster.


@atexit.register
def _hold_window():
    """Keep the final frame on screen until a keypress, so a run's end state is
    visible/recordable instead of the window vanishing when the program exits."""
    if _SHOW and _model is not None:
        try:
            print("[sim] done — click the window and press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass

_model = None
_data = None
_cam_onboard = None
_cam_chase = None
# dead-reckoned pose: metres (x, y), radians heading
_pose = {"x": 0.0, "y": 0.0, "heading": 0.0}


def _init():
    global _model, _data, _cam_onboard, _cam_chase
    _model = mujoco.MjModel.from_xml_string(_XML)
    _data = mujoco.MjData(_model)
    _cam_onboard = mujoco.Renderer(_model, height=480, width=640)
    _cam_chase = mujoco.Renderer(_model, height=480, width=640)
    _apply()


def _apply():
    """Write the dead-reckoned pose into the rover's free joint and refresh."""
    h = _pose["heading"]
    _data.qpos[:7] = [_pose["x"], _pose["y"], 0.12,
                      math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)]
    mujoco.mj_forward(_model, _data)


def _show(onboard_rgb=None):
    """Render the chase view (with onboard inset) into the demo window. Called on
    every frame AND every motion, so deterministic behaviors (e.g. bare 'turn
    left' that never call perception) still animate the window."""
    global _SHOW
    if not _SHOW:
        return
    try:
        if onboard_rgb is None:
            _cam_onboard.update_scene(_data, camera="onboard")
            onboard_rgb = _cam_onboard.render()
        _cam_chase.update_scene(_data, camera="chase")
        disp = _cam_chase.render().copy()
        inset = cv2.resize(onboard_rgb, (160, 120))
        disp[8:128, 8:168] = inset
        cv2.rectangle(disp, (8, 8), (168, 128), (255, 255, 255), 1)
        cv2.putText(disp, "onboard (Gemma sees)", (8, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("RoverCrew - MuJoCo", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    except cv2.error:
        _SHOW = False  # no display available; run headless


def get_frame() -> bytes:
    """Render the onboard camera (for the crew) and show a chase window."""
    if _model is None:
        _init()
    _cam_onboard.update_scene(_data, camera="onboard")
    rgb = _cam_onboard.render()  # HxWx3 RGB uint8
    _show(rgb)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("mujoco frame encode failed")
    return buf.tobytes()


def do_action(action: str):
    if _model is None:
        _init()
    x0, y0, h0 = _pose["x"], _pose["y"], _pose["heading"]
    step = config.SIM_STEP_M
    h1 = h0
    if action == "turn_left":
        h1 = h0 + config.SIM_TURN_RAD
        step *= 0.35          # turns advance a little (arc) so the rover rounds
    elif action == "turn_right":  # obstacles -- but not so much it wanders off
        h1 = h0 - config.SIM_TURN_RAD
        step *= 0.35
    elif action == "veer_left":   # gentle arc + FULL forward step: center on the
        h1 = h0 + config.SIM_TURN_RAD * 0.45   # target WHILE closing distance
    elif action == "veer_right":
        h1 = h0 - config.SIM_TURN_RAD * 0.45
    elif action != "forward":  # stop, done, back, unknown -> hold
        return
    x1 = x0 + step * math.cos(h1)
    y1 = y0 + step * math.sin(h1)
    # Keep the rover on the floor area. Bounds are wide enough for retreat /
    # go_around / avoid to drive well clear of the start, but not off the world.
    x1 = min(7.5, max(-4.5, x1))
    y1 = min(5.5, max(-5.5, y1))

    if _SHOW and _ANIM_FRAMES > 1:
        # Glide from old pose to new so the move is watchable on camera.
        for i in range(1, _ANIM_FRAMES + 1):
            a = i / _ANIM_FRAMES
            _pose["x"], _pose["y"] = x0 + (x1 - x0) * a, y0 + (y1 - y0) * a
            _pose["heading"] = h0 + (h1 - h0) * a
            _apply()
            _show()
            time.sleep(_ANIM_DT)
    else:
        _pose["x"], _pose["y"], _pose["heading"] = x1, y1, h1
        _apply()


def send_cmd(cmd: dict):
    # Only the emergency stop reaches here; nothing to integrate.
    pass

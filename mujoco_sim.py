"""Self-contained MuJoCo sim backend (USE_MJC=1).

Unlike the Cyberwave twin, this renders a real onboard camera, so the Gemma
crew actually *sees* the simulated world (a red target box) and can drive to it
and reach `done`. The rover is dead-reckoned (kinematic free joint) from the
crew's discrete actions. A chase-camera window (with the onboard view inset)
shows the robot moving for the demo recording.

    USE_MJC=1 ./.venv/bin/python main.py "red box"
"""
import math

import cv2
import mujoco
import numpy as np

import config

# Gravity off: we set the pose kinematically via mj_forward (no dynamics), so
# the rover holds exactly the pose we author each step.
_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.35 0.3"
             rgb2="0.25 0.3 0.25" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="1 -1 5" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="14 14 0.1" material="grid"/>
    <body name="rover" pos="0 0 0.12">
      <freejoint/>
      <geom type="box" size="0.22 0.16 0.10" rgba="0.2 0.45 0.95 1"/>
      <geom type="box" size="0.05 0.17 0.02" pos="0.18 0 0.06" rgba="0.1 0.1 0.1 1"/>
      <camera name="onboard" pos="0.22 0 0.16" xyaxes="0 -1 0 0 0 1"/>
    </body>

    <!-- TARGET: red cup, off to the left so the rover must search to find it -->
    <body name="target" pos="3.0 1.5 0.25">
      <geom type="cylinder" size="0.20 0.25" rgba="0.9 0.1 0.1 1"/>
    </body>

    <!-- DISTRACTORS: same shape, different colours -> Gemma must discriminate -->
    <body name="distractor_green" pos="2.6 0.0 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.15 0.75 0.2 1"/>
    </body>
    <body name="distractor_blue" pos="3.2 -2.4 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.15 0.3 0.9 1"/>
    </body>
    <body name="distractor_yellow" pos="1.4 1.7 0.22">
      <geom type="cylinder" size="0.16 0.22" rgba="0.9 0.8 0.1 1"/>
    </body>

    <camera name="chase" mode="trackcom" pos="2.0 -5.5 4.5"/>
  </worldbody>
</mujoco>
"""

import os

_SHOW = os.environ.get("MJC_WINDOW", "1") == "1"  # MJC_WINDOW=0 for headless

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


def get_frame() -> bytes:
    """Render the onboard camera (for the crew) and show a chase window."""
    if _model is None:
        _init()
    _cam_onboard.update_scene(_data, camera="onboard")
    rgb = _cam_onboard.render()  # HxWx3 RGB uint8

    # Demo window: chase view with the onboard view inset top-left.
    global _SHOW
    if _SHOW:
        try:
            _cam_chase.update_scene(_data, camera="chase")
            disp = _cam_chase.render().copy()
            inset = cv2.resize(rgb, (160, 120))
            disp[8:128, 8:168] = inset
            cv2.rectangle(disp, (8, 8), (168, 128), (255, 255, 255), 1)
            cv2.putText(disp, "onboard (Gemma sees)", (8, 144),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow("RoverCrew - MuJoCo", cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except cv2.error:
            _SHOW = False  # no display available; run headless

    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("mujoco frame encode failed")
    return buf.tobytes()


def do_action(action: str):
    if _model is None:
        _init()
    step = config.SIM_STEP_M
    if action == "turn_left":
        _pose["heading"] += config.SIM_TURN_RAD
        step *= 0.5            # turns also advance (arc), so the rover always
    elif action == "turn_right":  # closes distance instead of spinning in place
        _pose["heading"] -= config.SIM_TURN_RAD
        step *= 0.5
    elif action != "forward":  # stop, done, back, unknown -> hold
        return
    _pose["x"] += step * math.cos(_pose["heading"])
    _pose["y"] += step * math.sin(_pose["heading"])
    _apply()


def send_cmd(cmd: dict):
    # Only the emergency stop reaches here; nothing to integrate.
    pass

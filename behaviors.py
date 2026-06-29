"""Roro's behavior library: one control policy per spoken intent.

Hybrid design: Gemma (agents.perceive) reports WHAT/WHERE the target is each
step; deterministic code here decides the MOTION. Every behavior uses the proven
stop-look-move cadence -- pulse a motion, stop, let the camera settle, then
capture a sharp frame for the next decision (continuous driving blurs every
frame and perception goes blind).

    run({"intent": "retreat", "target": "orange case", "direction": None})

Intents: approach, retreat, go_around, turn, avoid, stop.
"""
import os
import time

import agents
import config
import rover_client as rover


def banner(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


# ---------- shared helpers ----------
def _look(target: str) -> dict:
    """Capture a (settled) frame and run Gemma perception on it."""
    return agents.perceive(rover.get_frame(), target)


def _pulse(action: str):
    """Stop-look-move: run one motion briefly, then stop and let the camera settle."""
    rover.do_action(action)
    time.sleep(config.MOVE_TIME)
    rover.send_cmd(config.cmd_stop())
    time.sleep(config.SETTLE_TIME)


def _safe(per: dict, action: str) -> str:
    """Run the safety gate; return the (possibly overridden) action."""
    safe = agents.safety_check(per, action)
    return action if safe.get("approved") else (safe.get("override") or "stop")


def _log(step: int, per: dict, action: str):
    print(f"[{step:03d}] see={per.get('target_visible')} "
          f"bearing={per.get('bearing')} dist={per.get('distance')} "
          f"obs={per.get('obstacle_ahead')} -> DO {action}")


# ---------- intent: approach (move toward X) ----------
def approach(target: str) -> bool:
    """Search for the target, face it, drive in, stop when near. (proven logic)"""
    search_dir = "turn_" + config.SEARCH_DIR
    for step in range(1, config.MAX_STEPS + 1):
        per = _look(target)
        visible, bearing, dist = (per.get("target_visible"),
                                  per.get("bearing"), per.get("distance"))
        if visible and bearing in ("left", "right"):
            search_dir = "turn_" + bearing           # remember which side it's on
        if not visible:
            action = search_dir                      # search toward last-seen side
        elif dist == "near":
            action = "done"                          # arrived (trust near here)
        elif bearing in ("left", "right"):
            action = "turn_" + bearing               # face it
        else:
            action = "forward"                       # centered + far/mid -> close in
        if action != "done":
            action = _safe(per, action)
        _log(step, per, action)
        if action == "done":
            rover.do_action("stop")
            banner(f"REACHED {target!r} in {step} steps 🎯")
            return True
        _pulse(action)
    banner(f"gave up after {config.MAX_STEPS} steps (no {target!r})")
    return False


# ---------- intent: retreat (move away from X) ----------
def retreat(target: str) -> bool:
    """Turn until the target is behind us (out of frame), then drive forward away.
    Front camera only -- we never reverse blind. Self-calibrating: rotates until
    the target leaves the view rather than trusting a fixed turn angle."""
    banner(f"RETREAT from {target!r}: turning away, then driving off")
    # phase 1: turn away until the target is no longer in view (cap ~270deg).
    for _ in range(3 * config.TURN_PULSES_90):
        per = _look(target)
        if not per.get("target_visible"):
            break
        _pulse("turn_left")
    # phase 2: drive forward away. Use ONLY the obstacle check -- don't let the
    # safety 'too close to target' veto stop us (the whole point is to leave it).
    fwd_steps = int(os.environ.get("RETREAT_STEPS") or 6)
    for step in range(1, fwd_steps + 1):
        per = _look(target)
        action = "turn_left" if per.get("obstacle_ahead") else "forward"
        _log(step, per, action)
        _pulse(action)
    rover.do_action("stop")
    banner(f"moved away from {target!r}")
    return True


# ---------- intent: go_around (pass around X) ----------
def go_around(target: str) -> bool:
    """Best-effort: approach to mid range, then arc a semicircle past the target."""
    side = (os.environ.get("AROUND_SIDE") or "left").lower()
    turn_to = "turn_" + side
    turn_back = "turn_right" if side == "left" else "turn_left"
    banner(f"GO AROUND {target!r} (via {side})")
    # phase 1: get within mid range of the target
    search_dir = "turn_" + config.SEARCH_DIR
    for step in range(1, config.MAX_STEPS + 1):
        per = _look(target)
        visible, bearing, dist = (per.get("target_visible"),
                                  per.get("bearing"), per.get("distance"))
        if visible and bearing in ("left", "right"):
            search_dir = "turn_" + bearing
        if visible and dist in ("mid", "near"):
            break
        action = search_dir if not visible else (
            "turn_" + bearing if bearing in ("left", "right") else "forward")
        action = _safe(per, action)
        _log(step, per, action)
        _pulse(action)
    # phase 2: arc around -- turn out, drive past, turn back, drive past
    for _ in range(config.TURN_PULSES_90):
        _pulse(turn_to)
    for _ in range(3):
        _pulse(_safe(_look(target), "forward"))
    for _ in range(config.TURN_PULSES_90):
        _pulse(turn_back)
    for _ in range(2):
        _pulse(_safe(_look(target), "forward"))
    rover.do_action("stop")
    banner(f"went around {target!r}")
    return True


# ---------- intent: turn (rotate, optionally to a landmark) ----------
def turn(direction: str | None, target: str) -> bool:
    if target:                                       # "turn left at the door"
        banner(f"TURN to face {target!r}")
        sdir = "turn_" + (direction or config.SEARCH_DIR)
        for step in range(1, config.MAX_STEPS + 1):
            per = _look(target)
            if per.get("target_visible") and per.get("bearing") == "center":
                rover.do_action("stop")
                banner(f"facing {target!r}")
                return True
            if per.get("target_visible") and per.get("bearing") in ("left", "right"):
                action = "turn_" + per["bearing"]
            else:
                action = sdir
            _log(step, per, action)
            _pulse(action)
        banner(f"could not center {target!r}")
        return False
    d = direction or "left"                           # bare "turn left"/"spin"
    banner(f"TURN {d} ~90")
    for _ in range(config.TURN_PULSES_90):
        _pulse("turn_" + d)
    rover.do_action("stop")
    return True


# ---------- intent: avoid (move while steering clear of X) ----------
def avoid(target: str) -> bool:
    """Drive forward; whenever the target appears close/ahead, steer away from it."""
    banner(f"AVOID {target!r} while moving")
    steps = int(os.environ.get("AVOID_STEPS") or 10)
    for step in range(1, steps + 1):
        per = _look(target)
        visible, bearing, dist = (per.get("target_visible"),
                                  per.get("bearing"), per.get("distance"))
        if visible and dist in ("near", "mid"):
            if bearing == "left":
                action = "turn_right"                # steer away from it
            elif bearing == "right":
                action = "turn_left"
            else:                                    # dead ahead -> veer to a side
                action = "turn_right" if config.SEARCH_DIR == "left" else "turn_left"
        else:
            action = "forward"
        action = _safe(per, action)
        _log(step, per, action)
        _pulse(action)
    rover.do_action("stop")
    banner(f"done avoiding {target!r}")
    return True


# ---------- dispatcher ----------
def run(cmd: dict):
    """Dispatch a parsed command dict to the matching behavior. Always stops safe."""
    intent = cmd.get("intent", "unknown")
    target = cmd.get("target", "")
    direction = cmd.get("direction")
    try:
        if intent == "approach":
            approach(target or "object")
        elif intent == "retreat":
            retreat(target or "object")
        elif intent == "go_around":
            go_around(target or "object")
        elif intent == "turn":
            turn(direction, target)
        elif intent == "avoid":
            avoid(target or "object")
        elif intent == "stop":
            rover.do_action("stop")
            banner("STOP")
        else:
            banner(f"unknown command: {cmd}")
            rover.do_action("stop")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        rover.send_cmd(config.cmd_stop())

"""Roro's behavior library: one control policy per spoken intent.

Hybrid design: Gemma (agents.perceive) reports WHAT/WHERE the target is each
step; deterministic code here decides the MOTION. Every behavior uses the proven
stop-look-move cadence -- pulse a motion, stop, let the camera settle, then
capture a sharp frame for the next decision (continuous driving blurs every
frame and perception goes blind).

Preemption: behaviors check a shared interrupt at every step boundary, so a new
voice command ("Roro, stop / move away") can abort the current behavior mid-run.
The supervisor (main.py) feeds new commands via set_pending() and picks up the
next one via take_pending().

    run({"intent": "retreat", "target": "orange case", "direction": None})

Intents: approach, retreat, go_around, turn, avoid, stop.
"""
import os
import threading
import time

import agents
import config
import rover_client as rover


def banner(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


# ---------- preemption: shared command slot + interrupt ----------
_interrupt = threading.Event()
_pending = None
_lock = threading.Lock()


def set_pending(steps):
    """Queue a new command (a list of step dicts) and signal the running behavior
    to abort so the supervisor can switch to it (thread-safe). Latest wins."""
    global _pending
    with _lock:
        _pending = steps
    _interrupt.set()


def take_pending():
    """Pop the queued step list (or None) and clear the interrupt."""
    global _pending
    with _lock:
        steps, _pending = _pending, None
    _interrupt.clear()
    return steps


def has_pending() -> bool:
    return _pending is not None


def aborted() -> bool:
    """True if a new command arrived -- behaviors poll this each step."""
    return _interrupt.is_set()


# ---------- shared helpers ----------
_last_ms = 0.0  # wall time of the most recent Gemma-4 vision call (Cerebras speed)


def _look(target: str) -> dict:
    """Capture a (settled) frame and run Gemma perception on it, timing the call."""
    global _last_ms
    t = time.time()
    per = agents.perceive(rover.get_frame(), target)
    _last_ms = (time.time() - t) * 1000
    return per


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
    print(f"[{step:03d}] {_last_ms:4.0f}ms gemma-4/cerebras | "
          f"see={per.get('target_visible')} "
          f"bearing={per.get('bearing')} dist={per.get('distance')} "
          f"obs={per.get('obstacle_ahead')} -> DO {action}")


# ---------- intent: approach (move toward X) ----------
def approach(target: str) -> str:
    """Search for the target, face it, drive in, stop when near. (proven logic)"""
    search_dir = "turn_" + config.SEARCH_DIR
    for step in range(1, config.MAX_STEPS + 1):
        if aborted():
            return "preempted"
        per = _look(target)
        visible, bearing, dist = (per.get("target_visible"),
                                  per.get("bearing"), per.get("distance"))
        if visible and bearing in ("left", "right"):
            search_dir = "turn_" + bearing           # remember which side it's on
        if not visible:
            action = search_dir                      # search toward last-seen side
        elif dist == "near" and bearing in ("left", "right"):
            action = "turn_" + bearing               # face it for the final approach
        elif dist == "near":
            # centered + near: nudge in close, then finish (~5-10cm vs ~20cm).
            for _ in range(config.CLOSE_STEPS):
                if aborted():
                    return "preempted"
                _pulse("forward")
            rover.do_action("stop")
            banner(f"REACHED {target!r} in {step} steps 🎯")
            return "done"
        elif bearing in ("left", "right"):
            action = "veer_" + bearing               # advance WHILE centering (no orbit)
        else:
            action = "forward"                       # centered + far/mid -> close in
        # Safety only gates collision-capable motion (forward/veer). In-place
        # turns and search can't drive into anything, and routing them through the
        # LLM veto froze the rover with "stop" while centering next to the target.
        if action in ("forward", "veer_left", "veer_right"):
            action = _safe(per, action)
        _log(step, per, action)
        _pulse(action)
    banner(f"gave up after {config.MAX_STEPS} steps (no {target!r})")
    return "done"


# ---------- intent: retreat (move away from X) ----------
def retreat(target: str) -> str:
    """Turn until the target is behind us (out of frame), then drive forward away.
    Front camera only -- we never reverse blind. Self-calibrating: rotates until
    the target leaves the view rather than trusting a fixed turn angle."""
    banner(f"RETREAT from {target!r}: turning away, then driving off")
    # phase 1: turn away until the target is no longer in view (cap ~270deg).
    for _ in range(3 * config.TURN_PULSES_90):
        if aborted():
            return "preempted"
        if not _look(target).get("target_visible"):
            break
        _pulse("turn_left")
    # phase 2: drive forward away. Use ONLY the obstacle check -- don't let the
    # safety 'too close to target' veto stop us (the whole point is to leave it).
    fwd_steps = int(os.environ.get("RETREAT_STEPS") or 6)
    for step in range(1, fwd_steps + 1):
        if aborted():
            return "preempted"
        per = _look(target)
        action = "turn_left" if per.get("obstacle_ahead") else "forward"
        _log(step, per, action)
        _pulse(action)
    rover.do_action("stop")
    banner(f"moved away from {target!r}")
    return "done"


# ---------- intent: go_around (pass around X) ----------
def go_around(target: str) -> str:
    """Best-effort: approach to mid range, then arc a semicircle past the target."""
    side = (os.environ.get("AROUND_SIDE") or "left").lower()
    turn_to = "turn_" + side
    turn_back = "turn_right" if side == "left" else "turn_left"
    banner(f"GO AROUND {target!r} (via {side})")
    # phase 1: get within mid range of the target
    search_dir = "turn_" + config.SEARCH_DIR
    found = False
    for step in range(1, config.MAX_STEPS + 1):
        if aborted():
            return "preempted"
        per = _look(target)
        visible, bearing, dist = (per.get("target_visible"),
                                  per.get("bearing"), per.get("distance"))
        if visible and bearing in ("left", "right"):
            search_dir = "turn_" + bearing
        if visible and dist in ("mid", "near"):
            found = True
            break
        action = search_dir if not visible else (
            "turn_" + bearing if bearing in ("left", "right") else "forward")
        action = _safe(per, action)
        _log(step, per, action)
        _pulse(action)
    if not found:
        # never located the target -- don't fake an arc around empty space.
        rover.do_action("stop")
        banner(f"could not find {target!r} to go around")
        return "done"
    # phase 2: arc around -- turn out, drive past, turn back, drive past
    for seq in ([turn_to] * config.TURN_PULSES_90 + ["forward"] * 3
                + [turn_back] * config.TURN_PULSES_90 + ["forward"] * 2):
        if aborted():
            return "preempted"
        _pulse(_safe(_look(target), seq) if seq == "forward" else seq)
    rover.do_action("stop")
    banner(f"went around {target!r}")
    return "done"


# ---------- intent: turn (rotate, optionally to a landmark) ----------
def turn(direction, target: str) -> str:
    if target:                                       # "turn left at the door"
        banner(f"TURN to face {target!r}")
        sdir = "turn_" + (direction or config.SEARCH_DIR)
        for step in range(1, config.MAX_STEPS + 1):
            if aborted():
                return "preempted"
            per = _look(target)
            if per.get("target_visible") and per.get("bearing") == "center":
                rover.do_action("stop")
                banner(f"facing {target!r}")
                return "done"
            if per.get("target_visible") and per.get("bearing") in ("left", "right"):
                action = "turn_" + per["bearing"]
            else:
                action = sdir
            _log(step, per, action)
            _pulse(action)
        banner(f"could not center {target!r}")
        return "done"
    d = direction or "left"                           # bare "turn left"/"spin"
    banner(f"TURN {d} ~90")
    for _ in range(config.TURN_PULSES_90):
        if aborted():
            return "preempted"
        _pulse("turn_" + d)
    rover.do_action("stop")
    return "done"


# ---------- intent: avoid (move while steering clear of X) ----------
def avoid(target: str) -> str:
    """Drive forward; whenever the target appears close/ahead, steer away from it."""
    banner(f"AVOID {target!r} while moving")
    steps = int(os.environ.get("AVOID_STEPS") or 10)
    for step in range(1, steps + 1):
        if aborted():
            return "preempted"
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
    return "done"


# ---------- dispatcher ----------
def run(cmd: dict) -> str:
    """Dispatch a parsed command to the matching behavior. Returns 'done' or
    'preempted' (a new command interrupted it). Always stops the wheels."""
    intent = cmd.get("intent", "unknown")
    target = cmd.get("target", "")
    direction = cmd.get("direction")
    try:
        if intent == "approach":
            return approach(target or "object")
        if intent == "retreat":
            return retreat(target or "object")
        if intent == "go_around":
            return go_around(target or "object")
        if intent == "turn":
            return turn(direction, target)
        if intent == "avoid":
            return avoid(target or "object")
        if intent == "stop":
            rover.do_action("stop")
            banner("STOP")
            return "done"
        banner(f"unknown command: {cmd}")
        rover.do_action("stop")
        return "done"
    finally:
        rover.send_cmd(config.cmd_stop())

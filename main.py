"""The control loop: frame -> perception -> planner -> safety -> motor command.
Runs the agent crew at ~LOOP_HZ. Cerebras speed = real-time embodied agents.

    # dev, no robot (laptop webcam, prints motor cmds):
    USE_MOCK=1 ./.venv/bin/python main.py "red cup"

    # real rover (set ROVER_HOST in config.py, pi_server.py running on the Pi):
    ./.venv/bin/python main.py "red cup"
"""
import os
import sys
import time

import agents
import config
import rover_client as rover


def banner(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "red cup"
    # Voice mode: `main.py voice` (or VOICE=1) -> speak the command, Whisper
    # transcribes, Gemma extracts the target ("go towards the orange case").
    if target == "voice" or os.environ.get("VOICE") == "1":
        import voice
        target = voice.get_target_by_voice() or "red cup"
    banner(f"MISSION: find the {target!r}")
    body = {"sim": "Cyberwave UGV Beast (digital twin)",
            "mjc": "MuJoCo simulation (onboard camera)",
            "mock": "laptop webcam (no motors)"}.get(rover.MODE, f"rover @ {config.ROVER_HOST}")
    print(f"brain : Cerebras · {config.MODEL} (multimodal)")
    print(f"agents: perception -> planner -> critic -> safety (4 Gemma-4 calls/loop)")
    print(f"body  : {body}")

    print("warming up model...")
    agents._create(
        model=config.MODEL,
        messages=[{"role": "user", "content": "ready?"}],
        max_tokens=5,
    )

    period = 1.0 / config.LOOP_HZ
    max_steps = int(os.environ.get("MAX_STEPS") or 60)
    # Search direction memory: once the target has been glimpsed on a side, keep
    # searching toward that side instead of blindly spinning the planner's default
    # way (which can sweep ~360deg the long way round). SEARCH_DIR sets the very
    # first guess before the target is ever seen.
    search_dir = "turn_" + (os.environ.get("SEARCH_DIR") or "right")
    # Min-approach: Gemma's distance sense is unreliable and over-reports 'near',
    # which makes the rover declare 'done' from across the room without moving.
    # Force at least this many forward approach steps while the target is visible
    # and centered before 'done' is honoured, so it physically closes the gap.
    min_approach = int(os.environ.get("MIN_APPROACH") or 3)
    approached = 0
    # Stop-look-move timing: how long to pulse each motion, then how long to let
    # the camera settle (sharp frame) before deciding again. Tune via env.
    move_time = float(os.environ.get("MOVE_TIME") or 0.35)
    settle_time = float(os.environ.get("SETTLE_TIME") or 0.35)
    step = 0
    try:
        while True:
            step += 1
            if step > max_steps:
                banner(f"stopped after {max_steps} steps (no '{target}' reached)")
                rover.do_action("stop")
                break
            t0 = time.time()

            jpg = rover.get_frame()

            # Multi-agent crew, each an independent Gemma-4 call on Cerebras.
            # Time the inference to show Cerebras speed (4 calls back to back).
            ti = time.time()
            per = agents.perceive(jpg, target)        # 1. multimodal perception
            pl = agents.plan(per, target)             # 2. planner
            crit = agents.critique(per, pl, target)   # 3. critic (goal alignment)
            action = crit["action"]
            safe = agents.safety_check(per, action)   # 4. safety veto
            infer_ms = (time.time() - ti) * 1000
            if not safe["approved"]:
                action = safe["override"]

            # Remember which side the target was last seen on, and steer the
            # search that way when it's out of frame (so we turn toward it, not
            # away). Only applies while searching (target not visible + turning).
            bearing = per.get("bearing")
            if per.get("target_visible") and bearing in ("left", "right"):
                search_dir = "turn_" + bearing
            if not per.get("target_visible") and action in ("turn_left", "turn_right"):
                action = search_dir

            # Refine 'done' ONLY while the target is not yet near. Once perception
            # says 'near', accept 'done' regardless of bearing -- forcing more
            # turns/forwards at that point makes the rover overshoot and orbit the
            # target (it loses it, re-searches, reacquires on the other side...).
            dist = per.get("distance")
            if action == "done" and per.get("target_visible") and dist in ("mid", "far"):
                if bearing in ("left", "right"):
                    action = "turn_" + bearing               # face it first
                elif approached < min_approach and not per.get("obstacle_ahead"):
                    action = "forward"                       # close the gap
            if action == "forward":
                approached += 1

            dt = time.time() - t0
            print(
                f"[{step:03d}] {infer_ms:4.0f}ms/4-agents | "
                f"see={per.get('target_visible')} bearing={per.get('bearing')} "
                f"dist={per.get('distance')} obs={per.get('obstacle_ahead')} "
                f"| plan={pl.get('action')} | critic={crit.get('action')}"
                f"{'*' if crit.get('changed') else ''} ({crit.get('reason','')}) "
                f"| safety={safe.get('reason')} -> DO {action}"
            )

            if action == "done":
                rover.do_action("stop")
                banner(f"REACHED {target!r} in {step} steps 🎯")
                break

            # Stop-look-move: pulse the action briefly, then STOP and let the
            # camera settle before the next loop captures its decision frame.
            # Driving continuously made every frame motion-blurred, so perception
            # reported target_visible=False mid-turn and the rover circled forever.
            rover.do_action(action)
            time.sleep(move_time)
            rover.send_cmd(config.cmd_stop())
            time.sleep(settle_time)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        rover.send_cmd(config.cmd_stop())


if __name__ == "__main__":
    main()

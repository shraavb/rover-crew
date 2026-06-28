"""The control loop: frame -> perception -> planner -> safety -> motor command.
Runs the agent crew at ~LOOP_HZ. Cerebras speed = real-time embodied agents.

    # dev, no robot (laptop webcam, prints motor cmds):
    USE_MOCK=1 ./.venv/bin/python main.py "red cup"

    # real rover (set ROVER_HOST in config.py, pi_server.py running on the Pi):
    ./.venv/bin/python main.py "red cup"
"""
import sys
import time

import agents
import config
import rover_client as rover


def banner(s):
    print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "red cup"
    banner(f"MISSION: find the {target!r}")

    print("warming up model...")
    agents._create(
        model=config.MODEL,
        messages=[{"role": "user", "content": "ready?"}],
        max_tokens=5,
    )

    period = 1.0 / config.LOOP_HZ
    step = 0
    try:
        while True:
            step += 1
            t0 = time.time()

            jpg = rover.get_frame()

            # perceive + plan merged into one call (per loop: 1 vision call + local safety)
            pl = agents.sense_plan(jpg, target)
            per = pl  # perception fields live in the same dict
            action = pl["action"]
            safe = agents.safety_check(per, action)
            if not safe["approved"]:
                action = safe["override"]

            dt = time.time() - t0
            print(
                f"[{step:03d}] {dt*1000:4.0f}ms | "
                f"see={per.get('target_visible')} bearing={per.get('bearing')} "
                f"dist={per.get('distance')} obs={per.get('obstacle_ahead')} "
                f"| plan={pl['action']} ({pl.get('reason','')}) "
                f"| safety={safe['reason']} -> DO {action}"
            )

            if action == "done":
                rover.do_action("stop")
                banner(f"REACHED {target!r} in {step} steps 🎯")
                break

            rover.do_action(action)

            time.sleep(max(0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        rover.send_cmd(config.cmd_stop())


if __name__ == "__main__":
    main()

"""Roro's entry point: always-on voice control with mid-motion preemption.

    # always-on wake word -- say "Roro <command>" anytime, even while moving:
    ./.venv/bin/python main.py voice
    ./.venv/bin/python main.py "orange case"     # start a command, still listening

    # dev without a robot: USE_MOCK=1 (laptop webcam) / USE_SIM=1 / USE_MJC=1

A background thread listens for "Roro ..." continuously; Gemma parses each
command into {intent, target, direction}; behaviors.py runs the matching policy.
A new command preempts the current behavior at the next step boundary (~1s).
Ctrl-C to quit.
"""
import sys
import threading
import time

import agents
import config
import behaviors
import rover_client as rover
import voice


def supervise(initial_steps, listen=voice.listen_loop):
    """Run a queue of steps in order; a new spoken command preempts the current
    step and replaces the remaining queue. initial_steps runs first (empty ->
    idle until the first command). `listen` is the listener loop (always-on or
    push-to-talk)."""
    stop_event = threading.Event()
    listener = threading.Thread(
        target=listen, args=(behaviors.set_pending, stop_event), daemon=True)
    listener.start()

    queue = list(initial_steps or [])
    try:
        while True:
            if not queue:                            # idle: wait for a command
                if behaviors.has_pending():
                    queue = list(behaviors.take_pending() or [])
                    continue
                time.sleep(0.2)
                continue
            step = queue.pop(0)
            if len(queue) >= 0:
                print(f"[supervisor] step: {step}  ({len(queue)} more queued)")
            status = behaviors.run(step)             # 'done' or 'preempted'
            if status == "preempted":                # new command replaces the rest
                queue = list(behaviors.take_pending() or [])
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        stop_event.set()
        rover.send_cmd(config.cmd_stop())


def main():
    arg = " ".join(sys.argv[1:]).strip() or "voice"

    body = {"sim": "Cyberwave UGV Beast (digital twin)",
            "mjc": "MuJoCo simulation (onboard camera)",
            "mock": "laptop webcam (no motors)"}.get(rover.MODE, f"rover @ {config.ROVER_HOST}")
    print(f"brain : Cerebras · {config.MODEL} (multimodal)")
    print(f"body  : {body}")

    print("warming up model...")
    agents._create(model=config.MODEL,
                   messages=[{"role": "user", "content": "ready?"}], max_tokens=5)

    # `voice` -> always-on wake word; `ptt` -> push-to-talk (Enter to start/stop).
    # Any other text is an initial command; the listener still runs so you can
    # interrupt it by voice.
    if arg in ("voice", "ptt"):
        listen = voice.ptt_loop if arg == "ptt" else voice.listen_loop
        initial = []
    else:
        listen = voice.listen_loop
        initial = voice.parse_command(arg)
        behaviors.banner(f"COMMAND: {initial}")
    supervise(initial, listen)


if __name__ == "__main__":
    main()

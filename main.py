"""Roro's entry point: turn a command (typed or spoken) into a behavior.

    # typed natural command (no mic):
    ./.venv/bin/python main.py "move away from the orange case"
    ./.venv/bin/python main.py "orange case"          # bare object -> approach

    # spoken command (Whisper -> Gemma intent):
    ./.venv/bin/python main.py voice

    # dev without a robot: USE_MOCK=1 (laptop webcam) / USE_SIM=1 / USE_MJC=1

Gemma parses the command into {intent, target, direction}; behaviors.py runs the
matching control policy (approach / retreat / go_around / turn / avoid).
"""
import sys

import agents
import config
import behaviors
import rover_client as rover
import voice


def main():
    arg = " ".join(sys.argv[1:]).strip()
    if not arg:
        arg = "orange case"

    body = {"sim": "Cyberwave UGV Beast (digital twin)",
            "mjc": "MuJoCo simulation (onboard camera)",
            "mock": "laptop webcam (no motors)"}.get(rover.MODE, f"rover @ {config.ROVER_HOST}")
    print(f"brain : Cerebras · {config.MODEL} (multimodal)")
    print(f"body  : {body}")

    print("warming up model...")
    agents._create(model=config.MODEL,
                   messages=[{"role": "user", "content": "ready?"}], max_tokens=5)

    # Voice mode speaks the command; otherwise the CLI text IS the command.
    if arg == "voice":
        cmd = voice.get_command_by_voice()
    else:
        cmd = voice.parse_command(arg)
    behaviors.banner(f"COMMAND: {cmd}")

    behaviors.run(cmd)


if __name__ == "__main__":
    main()

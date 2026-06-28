"""Pre-flight setup check for rover-crew. Run before a mission.

    ./.venv/bin/python check_setup.py          # full check (pings the Pi)
    ./.venv/bin/python check_setup.py --mock    # skip Pi, check laptop/webcam only

Read-only: GETs the camera frame but sends NO motor commands (won't move the
rover). Motor wiring is verified separately while you watch the rover.
"""
import os
import sys

import config

MOCK = "--mock" in sys.argv
PASS, FAIL = 0, 0


def check(label, ok, detail=""):
    global PASS, FAIL
    mark = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("=== rover-crew setup check ===\n")

    print("1. Brain (laptop)")
    check("CEREBRAS_API_KEY set", bool(os.environ.get("CEREBRAS_API_KEY")),
          "export CEREBRAS_API_KEY=... before running" if not os.environ.get("CEREBRAS_API_KEY") else "")
    deps_ok = True
    for mod in ("cerebras.cloud.sdk", "requests", "cv2"):
        try:
            __import__(mod)
        except ImportError:
            deps_ok = False
            check(f"import {mod}", False, "pip install -r requirements.txt")
    if deps_ok:
        check("python deps importable", True)
    print()

    if MOCK:
        print("2. Webcam (mock mode)")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ok, frame = cap.read()
            check("laptop webcam reads a frame", ok,
                  f"{frame.shape[1]}x{frame.shape[0]}" if ok else "no camera / permission denied")
            cap.release()
        except Exception as e:  # noqa: BLE001
            check("laptop webcam reads a frame", False, str(e))
    else:
        print("2. Body (Raspberry Pi)")
        host_set = check("config.ROVER_HOST not placeholder",
                         config.ROVER_HOST != "192.168.1.50",
                         f"still {config.ROVER_HOST} — set to the Pi's IP (hostname -I on Pi)"
                         if config.ROVER_HOST == "192.168.1.50" else config.ROVER_HOST)
        if host_set:
            import requests
            base = f"http://{config.ROVER_HOST}:{config.ROVER_PORT}"
            try:
                r = requests.get(f"{base}/frame", timeout=3)
                ct = r.headers.get("content-type", "")
                check("GET /frame returns a JPEG", r.status_code == 200 and "image" in ct,
                      f"status={r.status_code} type={ct or '?'} "
                      f"(503=camera not ready, conn refused=pi_server not running)")
            except Exception as e:  # noqa: BLE001
                check("GET /frame reachable", False,
                      f"{type(e).__name__}: is pi_server.py running on the Pi? firewall/WiFi?")
    print()

    print(f"=== {PASS} pass, {FAIL} fail ===")
    if FAIL == 0:
        print("Ready. Motor check separately: watch the rover, run a short mission.")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

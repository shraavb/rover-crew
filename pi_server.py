"""RUNS ON THE ROVER'S RASPBERRY PI (the 'body').

Exposes two endpoints over WiFi:
  GET  /frame  -> latest camera frame as JPEG
  POST /cmd    -> forwards a JSON motion command to the UGV serial sub-controller

Setup on the Pi:
    pip install flask opencv-python pyserial
    python3 pi_server.py
Then note the Pi's IP (`hostname -I`) and put it in config.ROVER_HOST on the laptop.

NOTE: verify SERIAL_PORT + baud against your Waveshare wiki.
Common ports: /dev/serial0, /dev/ttyAMA0, /dev/ttyS0. Baud usually 115200.
If you prefer the camera via picamera2, swap the capture section (see comment).
"""
import json
import threading
import time

import cv2
import serial
from flask import Flask, Response, request, jsonify

SERIAL_PORT = "/dev/ttyAMA0"   # ESP32 sub-controller (verified: streams T:1001 feedback). serial0->ttyAMA10 is dead.
BAUD = 115200
CAM_INDEX = 0                   # USB cam index; 0 is usually correct

app = Flask(__name__)

# ---- Serial (motor) ----
try:
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
    print(f"[serial] open {SERIAL_PORT}@{BAUD}")
except Exception as e:
    ser = None
    print(f"[serial] WARNING could not open {SERIAL_PORT}: {e} (running camera-only)")


def send_serial(cmd: dict):
    if ser is None:
        print("[serial] (no port) would send:", cmd)
        return
    line = (json.dumps(cmd) + "\n").encode()
    ser.write(line)


# ---- Camera (background grab so /frame is always fresh) ----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)  # V4L2 backend; auto-backend hangs on this USB cam
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG needed or open() blocks
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
_latest = {"jpg": None}


def grab_loop():
    while True:
        ok, frame = cap.read()
        if ok:
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                _latest["jpg"] = buf.tobytes()
        time.sleep(0.03)


threading.Thread(target=grab_loop, daemon=True).start()


@app.get("/frame")
def frame():
    jpg = _latest["jpg"]
    if jpg is None:
        return ("no frame yet", 503)
    return Response(jpg, mimetype="image/jpeg")


@app.post("/cmd")
def cmd():
    data = request.get_json(force=True)
    send_serial(data)
    return jsonify({"ok": True, "sent": data})


@app.post("/stop")
def stop():
    send_serial({"T": 1, "L": 0.0, "R": 0.0})
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

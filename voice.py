"""Voice command input: mic -> Whisper transcription -> Gemma target extraction.

    say:  "please go towards the orange case"
    ->    target = "orange case"

Push-to-talk: call listen() and it records until you press Enter, transcribes
locally with faster-whisper, then asks Gemma 4 (on Cerebras) for the target.

    from voice import get_target_by_voice
    target = get_target_by_voice()        # blocks, returns "orange case"

Standalone test:
    ./.venv/bin/python voice.py
"""
import json
import sys
import threading
import queue

import numpy as np
import sounddevice as sd

import config  # loads .env / env first
import agents  # reuse the Cerebras client + _json_call

SAMPLE_RATE = 16000          # Whisper wants 16 kHz mono
WHISPER_MODEL = "base.en"    # good speed/accuracy on CPU; override via env

_model = None


def _get_model():
    """Lazy-load faster-whisper (first call downloads the model, then cached)."""
    global _model
    if _model is None:
        import os
        from faster_whisper import WhisperModel
        name = os.environ.get("WHISPER_MODEL") or WHISPER_MODEL
        print(f"[voice] loading whisper '{name}' ...")
        _model = WhisperModel(name, device="cpu", compute_type="int8")
    return _model


def record_until_enter() -> np.ndarray:
    """Record mono 16 kHz audio from the default mic until the user hits Enter."""
    frames = queue.Queue()

    def cb(indata, _frames, _time, status):
        if status:
            print(f"[voice] {status}", file=sys.stderr)
        frames.put(indata.copy())

    print("[voice] 🎤 recording... press Enter to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=cb):
        input()  # block until Enter
    print("[voice] ...stopped, transcribing.")

    chunks = []
    while not frames.empty():
        chunks.append(frames.get())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    """Local Whisper transcription. Returns the recognized text."""
    if audio.size == 0:
        return ""
    segments, _ = _get_model().transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text for s in segments).strip()


INTENTS = ("approach", "retreat", "go_around", "turn", "avoid", "stop", "unknown")


def parse_command(text: str) -> dict:
    """Ask Gemma 4 to turn a spoken command into a structured intent for Roro.

    Returns {"intent", "target", "direction"} where intent is one of INTENTS,
    target is a short noun phrase ("" for a bare turn), direction is left/right
    for `turn` (else null). A bare object with no verb -> approach.
    """
    prompt = (
        "You parse spoken commands for a home rover robot named Roro into a "
        "structured intent.\n"
        "intent is one of: approach (move toward / go to / find), retreat (move "
        "away / back away from), go_around (go around / circle / pass), turn "
        "(turn / rotate / spin), avoid (avoid / stay away from while moving), "
        "stop (stop / halt), unknown.\n"
        "target = the object the command refers to, as a short lowercase noun "
        "phrase with no articles (e.g. 'orange case'); empty string if none.\n"
        "direction = 'left' or 'right' for a turn command, else null.\n"
        "A bare object with no verb (e.g. 'orange case') means approach.\n"
        "Examples:\n"
        '  "please move away from the orange case" -> {"intent":"retreat","target":"orange case","direction":null}\n'
        '  "go around the chair" -> {"intent":"go_around","target":"chair","direction":null}\n'
        '  "turn left at the door" -> {"intent":"turn","target":"door","direction":"left"}\n'
        '  "spin around" -> {"intent":"turn","target":"","direction":null}\n'
        '  "avoid the backpack" -> {"intent":"avoid","target":"backpack","direction":null}\n'
        '  "orange case" -> {"intent":"approach","target":"orange case","direction":null}\n'
        f'Command: "{text}"\n'
        'Reply ONLY JSON: {"intent": "...", "target": "...", "direction": "left"|"right"|null}'
    )
    out = agents._json_call([{"role": "user", "content": prompt}], max_tokens=60)
    intent = (out.get("intent") or "unknown").strip().lower()
    if intent not in INTENTS:
        intent = "unknown"
    direction = out.get("direction")
    if direction not in ("left", "right"):
        direction = None
    return {
        "intent": intent,
        "target": (out.get("target") or "").strip().lower(),
        "direction": direction,
    }


def get_command_by_voice() -> dict:
    """Full pipeline: record -> transcribe -> parse command. intent 'unknown' on failure."""
    audio = record_until_enter()
    text = transcribe(audio)
    if not text:
        print("[voice] heard nothing.")
        return {"intent": "unknown", "target": "", "direction": None}
    print(f"[voice] heard: {text!r}")
    cmd = parse_command(text)
    print(f"[voice] command: {cmd}")
    return cmd


if __name__ == "__main__":
    print("Voice command test. Speak after the prompt.")
    c = get_command_by_voice()
    print(f"\nFINAL COMMAND = {c}")

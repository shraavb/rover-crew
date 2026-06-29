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


def extract_target(text: str) -> str:
    """Ask Gemma 4 for the object the user wants the rover to go to."""
    prompt = (
        "A user spoke a command to a rover. Extract ONLY the target object it "
        "should drive to, as a short noun phrase (e.g. 'orange case', 'red cup'). "
        "Lowercase, no articles, no verbs.\n"
        f'Command: "{text}"\n'
        'Reply ONLY JSON: {"target": "<noun phrase>"}'
    )
    out = agents._json_call([{"role": "user", "content": prompt}], max_tokens=40)
    return (out.get("target") or "").strip()


def get_target_by_voice() -> str:
    """Full pipeline: record -> transcribe -> extract target. Returns "" on failure."""
    audio = record_until_enter()
    text = transcribe(audio)
    if not text:
        print("[voice] heard nothing.")
        return ""
    print(f"[voice] heard: {text!r}")
    target = extract_target(text)
    print(f"[voice] target: {target!r}")
    return target


if __name__ == "__main__":
    print("Voice command test. Speak after the prompt.")
    t = get_target_by_voice()
    print(f"\nFINAL TARGET = {t!r}")

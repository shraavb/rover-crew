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
import os
import threading
import queue

import numpy as np
import sounddevice as sd

import config  # loads .env / env first
import agents  # reuse the Cerebras client + _json_call

SAMPLE_RATE = 16000          # Whisper wants 16 kHz mono
WHISPER_MODEL = "base.en"    # good speed/accuracy on CPU; override via env
# How long each always-on listen window records. Longer = room for longer
# commands but slower preemption. Override with LISTEN_WINDOW=<seconds>.
LISTEN_WINDOW = float(os.environ.get("LISTEN_WINDOW") or 4.5)


def _mic_device():
    """Input device index for MIC=<name substring> (e.g. MIC='MacBook'); else the
    system default. Bluetooth headsets (AirPods/Powerbeats) often capture silence
    -- set MIC='MacBook Pro Microphone' if the listener hears nothing."""
    name = os.environ.get("MIC")
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            print(f"[voice] mic: {d['name']}")
            return i
    print(f"[voice] MIC {name!r} not found; using system default")
    return None


_MIC = _mic_device()

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
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=cb, device=_MIC):
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


def _clean_step(step: dict) -> dict:
    """Normalize one raw step dict into {intent, target, direction}."""
    intent = (step.get("intent") or "unknown").strip().lower()
    if intent not in INTENTS:
        intent = "unknown"
    direction = step.get("direction")
    if direction not in ("left", "right"):
        direction = None
    return {"intent": intent,
            "target": (step.get("target") or "").strip().lower(),
            "direction": direction}


def parse_command(text: str) -> list[dict]:
    """Ask Gemma 4 to turn a spoken command into an ORDERED list of steps for Roro.

    A compound command ("go around the basket and to the orange case") becomes
    multiple steps run in sequence. Each step is {intent, target, direction};
    intent in INTENTS, target a short noun phrase ("" if none), direction
    left/right for `turn`. A bare object with no verb -> approach. Always returns
    a non-empty list (a single command -> one step).
    """
    prompt = (
        "You parse spoken commands for a home rover robot named Roro into an "
        "ordered list of steps. Split compound commands (joined by 'and', 'then', "
        "commas) into separate steps IN ORDER.\n"
        "Each step's intent is one of: approach (move toward / go to / find), "
        "retreat (move away / back away from), go_around (go around / circle / "
        "pass), turn (turn / rotate / spin), avoid (avoid / stay away from while "
        "moving), stop (stop / halt), unknown.\n"
        "target = the object, a short lowercase noun phrase with no articles "
        "(e.g. 'orange case'); empty string if none. direction = 'left'/'right' "
        "for a turn, else null. A bare object with no verb means approach.\n"
        "Examples:\n"
        '  "go around the basket and towards the orange case" -> {"steps":['
        '{"intent":"go_around","target":"basket","direction":null},'
        '{"intent":"approach","target":"orange case","direction":null}]}\n'
        '  "please move away from the orange case" -> {"steps":['
        '{"intent":"retreat","target":"orange case","direction":null}]}\n'
        '  "turn left then find the door" -> {"steps":['
        '{"intent":"turn","target":"","direction":"left"},'
        '{"intent":"approach","target":"door","direction":null}]}\n'
        '  "orange case" -> {"steps":[{"intent":"approach","target":"orange case","direction":null}]}\n'
        f'Command: "{text}"\n'
        'Reply ONLY JSON: {"steps": [{"intent":"...","target":"...","direction":"left"|"right"|null}, ...]}'
    )
    out = agents._json_call([{"role": "user", "content": prompt}], max_tokens=160)
    steps = out.get("steps") or []
    cleaned = [_clean_step(s) for s in steps if isinstance(s, dict)]
    return cleaned or [{"intent": "unknown", "target": "", "direction": None}]


# Wake word. "robot" transcribes far more reliably than "roro" (heard as "roll").
# Override with WAKE_WORD=<word>. The extras catch common Whisper mishearings.
_WAKE = (os.environ.get("WAKE_WORD") or "robot").lower()
WAKE_WORDS = tuple(dict.fromkeys(
    (_WAKE, "robot", "robo", "row bot", "roboat", "hey robot")))


def _strip_wake(text: str) -> str:
    """Return the command part after the first wake word occurrence."""
    low = text.lower()
    best = -1
    wlen = 0
    for w in WAKE_WORDS:
        i = low.find(w)
        if i != -1 and (best == -1 or i < best):
            best, wlen = i, len(w)
    if best == -1:
        return text
    return text[best + wlen:].lstrip(" ,.!?").strip()


def _record_window(seconds: float) -> np.ndarray:
    """Record a fixed-length mono 16 kHz window from the chosen mic."""
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=_MIC)
    sd.wait()
    return audio.flatten()


def listen_loop(on_command, stop_event, window: float = None):
    """Always-on listener with pause-based end-of-utterance (so long/compound
    commands aren't truncated by a fixed window). Records short chunks, starts
    buffering when it hears speech, and finalizes the utterance after a short
    silence -- then transcribes the whole thing, checks for the wake word, parses
    it, and hands the steps to on_command. Runs until stop_event is set."""
    _get_model()  # warm the model before announcing readiness
    wake = WAKE_WORDS[0]
    chunk = 0.4                                            # seconds per read
    thresh = float(os.environ.get("VAD_THRESH") or 0.012)  # speech RMS threshold
    end_silence = float(os.environ.get("VAD_SILENCE") or 0.9)  # pause that ends it
    max_utt = float(os.environ.get("VAD_MAX") or 12.0)    # hard cap per utterance
    debug = os.environ.get("DEBUG_VOICE") == "1"
    print(f"[voice] 👂 always-on (speak naturally): say '{wake} <command>'")

    buf, speaking, silence, dur = [], False, 0.0, 0.0
    while not stop_event.is_set():
        a = _record_window(chunk)
        if stop_event.is_set():
            break
        rms = float(np.sqrt((a ** 2).mean())) if a.size else 0.0
        if rms >= thresh:                                 # speech
            buf.append(a)
            speaking, silence, dur = True, 0.0, dur + chunk
        elif speaking:                                    # trailing silence
            buf.append(a)
            silence += chunk
            dur += chunk
        if speaking and (silence >= end_silence or dur >= max_utt):
            audio = np.concatenate(buf)
            buf, speaking, silence, dur = [], False, 0.0, 0.0
            text = transcribe(audio)
            if not text:
                continue
            hit = any(w in text.lower() for w in WAKE_WORDS)
            if debug:
                print(f"[voice:debug] {text!r} {'(WAKE)' if hit else ''}")
            if not hit:
                continue
            cmd_text = _strip_wake(text)
            if not cmd_text:                              # wake word only
                continue
            print(f"[voice] heard: {text!r}")
            steps = parse_command(cmd_text)
            print(f"[voice] steps: {steps}")
            on_command(steps)


def get_command_by_voice() -> list[dict]:
    """Full pipeline: record -> transcribe -> parse into an ordered step list."""
    audio = record_until_enter()
    text = transcribe(audio)
    if not text:
        print("[voice] heard nothing.")
        return [{"intent": "unknown", "target": "", "direction": None}]
    print(f"[voice] heard: {text!r}")
    steps = parse_command(text)
    print(f"[voice] steps: {steps}")
    return steps


if __name__ == "__main__":
    print("Voice command test. Speak after the prompt.")
    c = get_command_by_voice()
    print(f"\nFINAL STEPS = {c}")

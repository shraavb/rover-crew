"""The agent crew. Each agent = one Gemma-4 call on Cerebras. Multimodal perception
+ planning + safety. Fast inference is what makes running 3 calls per control loop
viable in real time -- that's the Cerebras story.
"""
import base64
import json
import os

from cerebras.cloud.sdk import Cerebras

try:  # SDK exposes a typed 429 error; fall back to status-code sniffing if not.
    from cerebras.cloud.sdk import RateLimitError
except ImportError:  # pragma: no cover
    RateLimitError = None

import config
import ratelimit

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

_MAX_RETRIES = 5


def _img_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


def _is_rate_limit(exc: Exception) -> bool:
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True
    return getattr(exc, "status_code", None) == 429


def _retry_after(exc: Exception) -> float:
    """Seconds to wait per the server. Prefer Retry-After header, else 1s."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", {}) or {}
    for key in ("retry-after", "retry-after-ms"):
        val = headers.get(key)
        if val is None:
            continue
        try:
            secs = float(val)
            return secs / 1000.0 if key.endswith("-ms") else secs
        except (TypeError, ValueError):
            pass
    return 1.0


def _create(**kwargs):
    """Rate-limited, 429-retrying wrapper around the chat completion call."""
    for attempt in range(_MAX_RETRIES):
        ratelimit.limiter.acquire()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below if not a 429
            if not _is_rate_limit(exc):
                raise
            wait = _retry_after(exc)
            ratelimit.limiter.penalize(wait)
            print(f"  rate-limited (429), backing off {wait:.1f}s "
                  f"[retry {attempt + 1}/{_MAX_RETRIES}]")
    raise RuntimeError("rate limit: exhausted retries")


def _json_call(messages, max_tokens=300) -> dict:
    """Call Gemma and parse a JSON object from the reply (robust to code fences)."""
    resp = _create(
        model=config.MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    txt = resp.choices[0].message.content.strip()
    # strip ```json fences if present
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.startswith("json"):
            txt = txt[4:]
    # grab the outermost braces
    start, end = txt.find("{"), txt.rfind("}")
    return json.loads(txt[start : end + 1])


# ---------- Agent 1: Perception (multimodal) ----------
def perceive(jpg: bytes, target: str) -> dict:
    """Look at the camera frame. Where is the target? Obstacles?"""
    prompt = (
        f'You are the PERCEPTION agent on a small ground rover (camera ~20cm off floor). '
        f'Target object: "{target}".\n'
        "Report ONLY what you clearly see. Be conservative.\n"
        "- target_visible: true ONLY if you are confident the target is in frame.\n"
        "- bearing: which third of the frame the target is in (left/center/right), else none.\n"
        "- distance: near if it fills much of the frame, far if tiny, mid otherwise.\n"
        "- obstacle_ahead: true ONLY if a large object/wall is close and directly blocking "
        "forward motion within ~1 rover-length. A clear floor ahead is NOT an obstacle. "
        "Default false when unsure.\n"
        "Reply with ONLY a JSON object:\n"
        '{"target_visible": true/false, '
        '"bearing": "left"|"center"|"right"|"none", '
        '"distance": "near"|"mid"|"far"|"none", '
        '"obstacle_ahead": true/false, '
        '"scene": "<5-word description>"}'
    )
    return _json_call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _img_url(jpg)}},
                ],
            }
        ]
    )


# ---------- Agent 2: Planner ----------
def plan(perception: dict, target: str) -> dict:
    """Decide the next high-level action from the perception report."""
    prompt = (
        f'You are the PLANNER agent. Goal: drive the rover to the "{target}".\n'
        f"Perception report: {json.dumps(perception)}\n"
        "Rules: if target not visible, turn to search. If visible & off-center, "
        "turn toward its bearing. If visible & centered & not near, go forward. "
        "If visible, centered, and near -> done.\n"
        'Reply ONLY JSON: {"action": "forward"|"turn_left"|"turn_right"|"stop"|"done", '
        '"reason": "<8-word reason>"}'
    )
    return _json_call([{"role": "user", "content": prompt}])


# ---------- Agent 1+2 merged: Perceive + Plan in one multimodal call ----------
def sense_plan(jpg: bytes, target: str) -> dict:
    """One multimodal call: look, report, decide. Halves API calls per loop vs
    separate perceive+plan, so the loop runs ~2x faster under the rate limit.
    Returns the perception fields AND the chosen action in one JSON object.
    """
    prompt = (
        f'You are the rover\'s vision+planning agent (camera ~20cm off floor). '
        f'Target object: "{target}".\n'
        "FIRST look and report ONLY what you clearly see (be conservative):\n"
        "- target_visible: true ONLY if confident the target is in frame.\n"
        "- bearing: which third of frame the target is in (left/center/right), else none.\n"
        "- distance: near if it fills much of frame, far if tiny, mid otherwise.\n"
        "- obstacle_ahead: true ONLY if a large object/wall is close and directly "
        "blocking forward motion within ~1 rover-length. Clear floor is NOT an "
        "obstacle. Default false when unsure.\n"
        "THEN decide the next action from what you saw:\n"
        "- target not visible -> turn to search.\n"
        "- visible & off-center -> turn toward its bearing.\n"
        "- visible & centered & not near -> forward.\n"
        "- visible & centered & near -> done.\n"
        "Reply with ONLY a JSON object:\n"
        '{"target_visible": true/false, '
        '"bearing": "left"|"center"|"right"|"none", '
        '"distance": "near"|"mid"|"far"|"none", '
        '"obstacle_ahead": true/false, '
        '"scene": "<5-word description>", '
        '"action": "forward"|"turn_left"|"turn_right"|"stop"|"done", '
        '"reason": "<8-word reason>"}'
    )
    return _json_call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _img_url(jpg)}},
                ],
            }
        ]
    )


# ---------- Agent 3: Safety ----------
def safety_check(perception: dict, proposed_action: str) -> dict:
    """Veto unsafe moves (drive forward into an obstacle)."""
    if proposed_action == "forward" and perception.get("obstacle_ahead"):
        return {"approved": False, "override": "turn_left", "reason": "obstacle ahead"}
    return {"approved": True, "override": None, "reason": "clear"}

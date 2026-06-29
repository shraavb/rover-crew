"""Side-by-side latency: Gemma 4 on Cerebras vs a GPU multimodal provider.

Sends the SAME perception prompt + image to both and reports per-call latency,
so the demo can show Cerebras' speed against a GPU baseline with real numbers.

Cerebras uses CEREBRAS_API_KEY + config.MODEL (already in .env).
The GPU side is optional and uses any OpenAI-compatible vision endpoint:
    GPU_API_KEY    (required to run the GPU column)
    GPU_BASE_URL   default https://api.openai.com/v1
    GPU_MODEL      default gpt-4o

    ./.venv/bin/python speed_compare.py [image.jpg] [runs]
    GPU_API_KEY=sk-... ./.venv/bin/python speed_compare.py test.jpg 5
"""
import base64
import os
import statistics
import sys
import time

import requests

import config
import agents

PROMPT = ('Look at this image. Is there an orange case? Reply ONLY JSON '
          '{"visible": true/false, "bearing": "left"|"center"|"right"|"none"}.')


def _img_data_url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()


def _content(url):
    return [{"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": url}}]


def time_cerebras(url, runs):
    msgs = [{"role": "user", "content": _content(url)}]
    agents.client.chat.completions.create(model=config.MODEL, messages=msgs, max_tokens=50)  # warm
    ts = []
    for _ in range(runs):
        t = time.time()
        agents.client.chat.completions.create(model=config.MODEL, messages=msgs, max_tokens=50)
        ts.append((time.time() - t) * 1000)
    return ts


def time_gpu(url, runs):
    key = os.environ.get("GPU_API_KEY")
    if not key:
        return None
    base = os.environ.get("GPU_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("GPU_MODEL", "gpt-4o")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": 50,
            "messages": [{"role": "user", "content": _content(url)}]}
    requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=60)  # warm
    ts = []
    for _ in range(runs):
        t = time.time()
        r = requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=60)
        r.raise_for_status()
        ts.append((time.time() - t) * 1000)
    return ts


def report(name, ts):
    if not ts:
        print(f"  {name:30} (skipped)")
        return None
    med = statistics.median(ts)
    print(f"  {name:30} median {med:6.0f} ms   (min {min(ts):.0f}, max {max(ts):.0f}, n={len(ts)})")
    return med


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    url = _img_data_url(img)
    print(f"\nMultimodal latency  (image={img}, runs={runs})\n")
    c = report(f"Cerebras / {config.MODEL}", time_cerebras(url, runs))
    g = report(f"GPU / {os.environ.get('GPU_MODEL', 'gpt-4o')}", time_gpu(url, runs))
    print()
    if c and g:
        print(f"  => Cerebras is {g / c:.1f}x faster per multimodal call.\n")
    elif c:
        print("  => Set GPU_API_KEY (+ optional GPU_BASE_URL / GPU_MODEL) to add the "
              "GPU column.\n")

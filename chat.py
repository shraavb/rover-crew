"""Minimal streaming chat against Cerebras. Set MODEL to the Gemma 4 id from list_models.py."""
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

MODEL = os.environ.get("MODEL", "gemma-4-31b")

stream = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a fast, terse assistant."},
        {"role": "user", "content": "In one sentence, why is Cerebras inference fast?"},
    ],
    stream=True,
    max_tokens=200,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()

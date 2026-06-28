"""HOUR-0 RISK TEST: does Cerebras Gemma-4-31b accept image input?
Run this FIRST. If it prints a sane description of the image, multimodal works
and the whole project is viable. If it errors on the image, we pivot (see notes).

Usage:
    export CEREBRAS_API_KEY=csk-...
    ./.venv/bin/python vision_test.py path/to/any.jpg
"""
import base64
import sys
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])
MODEL = os.environ.get("MODEL", "gemma-4-31b")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def main():
    if len(sys.argv) < 2:
        print("usage: python vision_test.py <image.jpg>")
        sys.exit(1)

    data_url = encode_image(sys.argv[1])

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence. List any objects you see."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()

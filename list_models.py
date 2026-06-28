"""List available models — find the exact Gemma 4 model id."""
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

models = client.models.list()
for m in models.data:
    print(m.id)

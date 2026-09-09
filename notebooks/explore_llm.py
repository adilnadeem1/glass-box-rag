from groq import Groq
from dotenv import load_dotenv
import yaml
import os

load_dotenv() # reads .env and populates os.environ

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

model_name = config["llm"]["model_name"]

# Fail fast with a clear message instead of a raw 404 traceback
available_models = [m.id for m in client.models.list().data]
if model_name not in available_models:
    raise ValueError(
        f"'{model_name}' is not available on your Groq account.\n"
        f"Available models:\n" + "\n".join(f"  - {m}" for m in available_models)
    )

response = client.chat.completions.create(
    model=model_name,
    temperature=config["llm"]["temperature"],
    max_tokens=config["llm"]["max_tokens"],
    messages=[
        { "role": "user", "content": "Say hello in exactly five words." }
    ]
)

print(response.choices[0].message.content)

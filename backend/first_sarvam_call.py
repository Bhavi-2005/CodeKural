"""Make one minimal Sarvam chat completion to verify local API credentials."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("SARVAM_API_KEY is missing from the project .env file.")

client = SarvamAI(api_subscription_key=api_key, timeout=30.0)
response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[
        {
            "role": "user",
            "content": "In one sentence, why should code validate user input?",
        }
    ],
    max_tokens=120,
    reasoning_effort=None,
    request_options={"max_retries": 1},
)

answer = response.choices[0].message.content
if not answer:
    raise SystemExit("Sarvam returned an empty chat response.")
print(answer)

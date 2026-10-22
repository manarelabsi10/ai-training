import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

config = types.GenerateContentConfig(
    temperature=0.3,
    max_output_tokens=1000,
    system_instruction=(
        "You are a summarizer. Summarize the text the user sends "
        "in 3 to 5 short sentences. Keep the main ideas only."
    ),
)


def summarize(text):
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=text,
                config=config,
            )
            return response.text
        except errors.ServerError:
            print(f"Server busy, retry {attempt + 1}/3...")
            time.sleep(5)
    return "Still failing, try again later."


def read_text():
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


print("Paste your text, then type END on a new line. Type 'exit' to quit.")

while True:
    print("\nText:")
    text = read_text()
    if text.strip().lower() in ("exit", "quit"):
        break
    print("Summary:", summarize(text))
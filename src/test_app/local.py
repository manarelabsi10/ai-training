import ollama

model = "gemma3:1b"

system_prompt = "You are a summarizer. Summarize the text in 3 to 5 short sentences."


def summarize(text):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.3, "num_predict": 300},
    )
    return response["message"]["content"]


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
    if not text.strip():
        print("Empty text, paste something first.")
        continue
    print("Summary:", summarize(text))
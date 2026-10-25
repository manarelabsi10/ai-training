import ollama

PROMPT = "You are a summarizer. Summarize the text in 1 to 3 short sentences."


def summarize(text):
    response = ollama.chat(
        model="gemma3:1b",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0.3, "num_predict": 300},
    )
    return response["message"]["content"]


print("enter your text...")
while True:
    lines = []
    while (line := input()) != "END":
        lines.append(line)
    text = "\n".join(lines).strip()
    if text:
        print("Summary:", summarize(text), "\n")

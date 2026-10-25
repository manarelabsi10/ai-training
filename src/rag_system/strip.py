import json

with open("embeddings.json", encoding="utf-8") as f:
    data = json.load(f)

# keep only page and text
data = [{"page": d["page"], "text": d["text"]} for d in data]

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("done:", len(data), "chunks")
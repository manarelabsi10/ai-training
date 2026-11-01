from Speculative_RAG import retrieve, cluster, build_subsets, draft

QUESTION = "What is the name of the model used for virtual try-on?"

docs, _metas, embs = retrieve(QUESTION, k=10)
labels = cluster(embs, n_clusters=3)
subsets = build_subsets(docs, labels, n_subsets=3)

lines = []

lines.append("=== ALL 10 RETRIEVED CHUNKS ===")
for rank, d in enumerate(docs, start=1):
    has_name = "LaDI" in d
    has_label = "Model 3" in d
    lines.append(f"\n[rank {rank}] has LaDI-VTON: {has_name} | has 'Model 3': {has_label}")
    lines.append(d)

for i, subset in enumerate(subsets, start=1):
    lines.append(f"\n\n=== SUBSET {i} ({len(subset)} chunks) ===")
    for j, d in enumerate(subset, start=1):
        lines.append(f"\n[{j}] has LaDI-VTON: {'LaDI' in d} | has 'Model 3': {'Model 3' in d}")
        lines.append(d)
    result = draft(QUESTION, subset)
    lines.append(f"\n--> DRAFT ANSWER: {result['answer']}")
    lines.append(f"--> DRAFT RATIONALE: {result['rationale']}")

text = "\n".join(lines)
print(text)

with open("debug_chunks_output.txt", "w", encoding="utf-8") as f:
    f.write(text)
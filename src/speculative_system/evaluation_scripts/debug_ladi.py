from Speculative_RAG import get_chroma_collection, get_embeddings

QUESTION = "What is the name of the model used for virtual try-on?"
col = get_chroma_collection()

# 1) how many chunks mention LaDI in the whole index?
all_docs = col.get(include=["documents"])["documents"]
hits = [d for d in all_docs if "LaDI" in d]
print("chunks containing LaDI in the whole index:", len(hits))

# 2) at what rank does each one show up for this question?
q = get_embeddings([QUESTION])[0]
res = col.query(query_embeddings=[q], n_results=50, include=["documents"])
for rank, d in enumerate(res["documents"][0], start=1):
    if "LaDI" in d:
        print(f"\n--- rank {rank} ---")
        print(d)
"""
Step 4: Given a question, find matching CHILD chunks, then return their
unique PARENT chunks -- so retrieval stays precise (small chunks matched
by meaning) but Gemini gets fuller context to answer from.
"""

from embed_store import get_embedding_model, get_chroma_collection, _load_parents_store


def retrieve(query: str, top_k_children: int = 8, max_parents: int = 5,
             model=None, collection=None, parents_store=None) -> list:
    """
    Returns a list of unique parent dicts:
        {"text": ..., "source": ..., "section": ..., "distance": ...}
    "distance" is the best (lowest) distance among that parent's matching
    children. Sorted closest-first, capped at max_parents.
    """
    if model is None:
        model = get_embedding_model()
    if collection is None:
        collection = get_chroma_collection()
    if parents_store is None:
        parents_store = _load_parents_store()

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k_children,
    )

    # Walk matching children, keep the BEST distance seen per parent_id
    best_distance_per_parent = {}
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        parent_id = metadata["parent_id"]
        if parent_id not in best_distance_per_parent or distance < best_distance_per_parent[parent_id]:
            best_distance_per_parent[parent_id] = distance

    # Sort parents by their best child distance, closest first
    ranked_parent_ids = sorted(best_distance_per_parent, key=best_distance_per_parent.get)

    retrieved = []
    for parent_id in ranked_parent_ids[:max_parents]:
        parent = parents_store.get(parent_id)
        if parent is None:
            continue  # shouldn't happen, but don't crash retrieval over it
        retrieved.append({
            "text": parent["text"],
            "source": parent["source"],
            "section": parent["section"],
            "distance": best_distance_per_parent[parent_id],
        })

    return retrieved


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question"')
    else:
        query = sys.argv[1]
        results = retrieve(query)
        print(f"Top {len(results)} parent chunks for: \"{query}\"\n")
        for i, r in enumerate(results):
            print(f"[{i}] {r['source']} ({r['section']}) -- best child distance={r['distance']:.3f}")
            print(r["text"][:250].replace("\n", " ") + "...\n")

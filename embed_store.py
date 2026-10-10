"""
Step 3: Embed CHILD chunks and store them in Chroma for search, while
saving PARENT chunks to a local JSON lookup file (they're not searched
directly, only fetched by ID once a child matches during retrieval).
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "kavao_research_children"
PARENTS_STORE_PATH = "./parents_store.json"


def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _load_parents_store() -> dict:
    path = Path(PARENTS_STORE_PATH)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_parents_store(store: dict):
    with open(PARENTS_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)


def add_paper_to_store(chunked_paper: dict, model=None, collection=None):
    """
    chunked_paper: the dict returned by chunk.py's chunk_paper(), i.e.
        {"parents": [...], "children": [...]}

    Embeds and stores children in Chroma; appends parents to the local
    JSON lookup file (keyed by parent id).
    """
    if model is None:
        model = get_embedding_model()
    if collection is None:
        collection = get_chroma_collection()

    parents = chunked_paper["parents"]
    children = chunked_paper["children"]

    # --- Save parents to the lookup file ---
    parents_store = _load_parents_store()
    for p in parents:
        parents_store[p["id"]] = {
            "text": p["text"],
            "source": p["source"],
            "section": p["section"],
        }
    _save_parents_store(parents_store)

    # --- Embed and store children in Chroma ---
    if not children:
        return 0, len(parents)

    texts = [c["text"] for c in children]
    embeddings = model.encode(texts).tolist()

    ids = [f"{c['source']}::{c['section']}::{c['parent_id']}::{i}" for i, c in enumerate(children)]
    metadatas = [
        {"source": c["source"], "section": c["section"], "parent_id": c["parent_id"]}
        for c in children
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(children), len(parents)


if __name__ == "__main__":
    import sys
    from extract import process_paper
    from chunk import chunk_paper

    papers_dir = Path(__file__).parent.parent / "papers"
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {papers_dir}. Add some and re-run.")
        sys.exit(0)

    model = get_embedding_model()
    collection = get_chroma_collection()

    total_children = 0
    total_parents = 0
    for pdf_path in pdf_files:
        processed = process_paper(str(pdf_path))
        chunked = chunk_paper(processed)
        n_children, n_parents = add_paper_to_store(chunked, model=model, collection=collection)
        total_children += n_children
        total_parents += n_parents
        print(f"{pdf_path.name}: {n_parents} parents, {n_children} children added")

    print(f"\nTotal: {total_parents} parents, {total_children} children stored")
    print(f"Vector DB: {CHROMA_DB_PATH}")
    print(f"Parents lookup file: {PARENTS_STORE_PATH}")

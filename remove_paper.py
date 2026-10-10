"""
Remove a paper's chunks from the vector store and parents lookup file,
by its source filename (e.g. "paper005.pdf"). Use this when you've
already run embed_store.py on a paper you now want to drop from the
corpus -- just deleting the PDF itself does NOT remove its stored data.

Usage: python src/remove_paper.py paper005.pdf
"""

import sys
from embed_store import get_chroma_collection, _load_parents_store, _save_parents_store


def remove_paper(source_filename: str):
    collection = get_chroma_collection()

    # --- Remove children (Chroma) ---
    collection.delete(where={"source": source_filename})

    # --- Remove parents (local JSON lookup) ---
    parents_store = _load_parents_store()
    before = len(parents_store)
    parents_store = {
        pid: p for pid, p in parents_store.items() if p["source"] != source_filename
    }
    removed_parents = before - len(parents_store)
    _save_parents_store(parents_store)

    print(f"Removed all children for '{source_filename}' from the vector store.")
    print(f"Removed {removed_parents} parent chunk(s) for '{source_filename}'.")
    if removed_parents == 0:
        print("(No parents matched that filename -- check it's spelled exactly as stored, e.g. 'paper005.pdf')")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/remove_paper.py <filename.pdf>")
        print("Example: python src/remove_paper.py paper005.pdf")
    else:
        remove_paper(sys.argv[1])

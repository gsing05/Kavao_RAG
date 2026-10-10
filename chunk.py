"""
Step 2: Turn a paper's sections into parent-child chunk pairs.

Parent-child (auto-merging) chunking: small "child" chunks are what gets
embedded and searched against, so retrieval stays precise. But once a
child matches, we hand Gemini its full "parent" chunk (the surrounding
section/paragraph group) instead of the isolated fragment -- this gives
the LLM real context to work with instead of a sentence floating alone.

Each parent gets a stable ID so the vector store can record which
parent a given child belongs to.
"""

import hashlib

MAX_PARENT_SIZE = 3000  # characters -- roughly a full section
MAX_CHILD_SIZE = 400    # characters -- roughly a paragraph or two
CHILD_OVERLAP = 50      # characters, used only in the fixed-size fallback


def _make_parent_id(source: str, section: str, index: int) -> str:
    """Stable, short ID for a parent chunk, unique per paper+section+position."""
    raw = f"{source}::{section}::{index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _fixed_size_split(text: str, max_size: int, overlap: int) -> list:
    pieces = []
    start = 0
    while start < len(text):
        end = start + max_size
        pieces.append(text[start:end])
        start += max_size - overlap
    return pieces


def _split_oversized_paragraphs(paragraphs: list, max_size: int) -> list:
    """
    Some PDFs extract with no blank-line breaks at all, so a whole section
    can come through as a single giant "paragraph". Without this, that
    paragraph would become one oversized parent chunk, defeating the
    point of chunking. Any paragraph already bigger than max_size gets
    pre-split with the fixed-size fallback before grouping happens.
    """
    result = []
    for para in paragraphs:
        if len(para) <= max_size:
            result.append(para)
        else:
            result.extend(_fixed_size_split(para, max_size, overlap=0))
    return result


def _group_paragraphs_into_parents(paragraphs: list, max_size: int) -> list:
    """
    Greedily group consecutive paragraphs into parent-sized blocks, so a
    parent is close to (but not over) max_size, without cutting a
    paragraph in half.
    """
    paragraphs = _split_oversized_paragraphs(paragraphs, max_size)
    parents = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) > max_size:
            parents.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        parents.append("\n\n".join(current))

    return parents


def chunk_paper(processed_paper: dict, max_parent: int = MAX_PARENT_SIZE,
                max_child: int = MAX_CHILD_SIZE, overlap: int = CHILD_OVERLAP) -> dict:
    """
    Turn {"source": ..., "sections": {name: text}} into parent and child chunks.

    Returns:
        {
            "parents": [{"id": ..., "text": ..., "source": ..., "section": ...}, ...],
            "children": [{"text": ..., "source": ..., "section": ..., "parent_id": ...}, ...]
        }
    """
    source = processed_paper["source"]
    parents = []
    children = []

    for section_name, section_text in processed_paper["sections"].items():
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        parent_texts = _group_paragraphs_into_parents(paragraphs, max_parent)

        for p_index, parent_text in enumerate(parent_texts):
            parent_id = _make_parent_id(source, section_name, p_index)
            parents.append({
                "id": parent_id,
                "text": parent_text,
                "source": source,
                "section": section_name,
            })

            child_paragraphs = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
            for para in child_paragraphs:
                if len(para) <= max_child:
                    children.append({
                        "text": para,
                        "source": source,
                        "section": section_name,
                        "parent_id": parent_id,
                    })
                else:
                    for piece in _fixed_size_split(para, max_child, overlap):
                        children.append({
                            "text": piece,
                            "source": source,
                            "section": section_name,
                            "parent_id": parent_id,
                        })

    return {"parents": parents, "children": children}


if __name__ == "__main__":
    import sys
    from extract import process_paper

    if len(sys.argv) < 2:
        print("Usage: python src/chunk.py <path_to_pdf>")
    else:
        processed = process_paper(sys.argv[1])
        result = chunk_paper(processed)
        print(f"{len(result['parents'])} parents, {len(result['children'])} children "
              f"produced from {processed['source']}")
        for p in result["parents"][:3]:
            print(f"\nPARENT [{p['id']}] section={p['section']} len={len(p['text'])}")
            print(p["text"][:150].replace("\n", " ") + "...")
        for c in result["children"][:3]:
            print(f"\n  CHILD -> parent={c['parent_id']} len={len(c['text'])}")
            print("  " + c["text"][:100].replace("\n", " ") + "...")

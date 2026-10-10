"""
Ties retrieval + generation into a single function: ask_kavao(question).
This is what app.py (the web wrapper) and the eval script both call.
"""

from retrieve import retrieve
from generate import generate_answer


def ask_kavao(question: str, max_parents: int = 5) -> dict:
    """
    Full RAG call: retrieve relevant parent chunks (via child-level search),
    generate a grounded answer. Returns both the answer and the sources
    used, so the web UI can show "answer" plus a "sources" list for
    transparency.
    """
    chunks = retrieve(question, max_parents=max_parents)

    if not chunks:
        return {
            "answer": "No matching research found in the corpus yet.",
            "sources": [],
        }

    answer = generate_answer(question, chunks)
    sources = [{"source": c["source"], "section": c["section"]} for c in chunks]

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python src/pipeline.py "your question"')
    else:
        result = ask_kavao(sys.argv[1])
        print(f"ANSWER:\n{result['answer']}\n")
        print("SOURCES USED:")
        for s in result["sources"]:
            print(f"  - {s['source']} ({s['section']})")

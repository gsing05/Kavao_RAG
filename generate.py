"""
Step 5: Take retrieved chunks + the user's question, ask Gemini to answer
using ONLY that text, and cite which source it came from.

This is the step that makes the tool trustworthy rather than just another
chatbot: the prompt explicitly forbids answering from Gemini's general
knowledge and requires every claim to be traceable to a retrieved chunk.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# transport="rest" avoids a known issue on some Windows setups where the
# default gRPC connection silently hangs (often due to antivirus/firewall
# interference) instead of returning a response or a clear error.
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"), transport="rest")

MODEL_NAME = "gemini-flash-latest"  # self-updating alias -- always points to Google's
                                     # current stable Flash model, so this stays
                                     # current without needing manual updates later

SYSTEM_INSTRUCTIONS = """You are a research assistant for Kavao, an Appalachian food-security \
social enterprise. You answer questions using ONLY the research excerpts provided below \
— never your own general knowledge. For every claim, cite the source paper it came from.

Rules:
- If the provided excerpts don't contain enough information to answer, say so plainly. \
Do not guess or fill gaps with outside knowledge.
- Keep answers grounded and specific. Quote or closely paraphrase the excerpt, don't embellish.
- Cite sources inline like this: (Source: <filename>, <section>)
"""


def build_prompt(question: str, retrieved_chunks: list) -> str:
    excerpts = "\n\n".join(
        f"[Excerpt {i+1} — Source: {c['source']}, Section: {c['section']}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return f"""{SYSTEM_INSTRUCTIONS}

RESEARCH EXCERPTS:
{excerpts}

QUESTION: {question}

ANSWER (with inline citations):"""


def generate_answer(question: str, retrieved_chunks: list) -> str:
    """
    Calls Gemini with the citation-grounded prompt and returns the answer text.
    Swapping models later (e.g. to Claude) means only touching this function.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = build_prompt(question, retrieved_chunks)
    response = model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    # Quick manual check: python src/generate.py "your question here"
    import sys
    from retrieve import retrieve

    if len(sys.argv) < 2:
        print('Usage: python src/generate.py "your question"')
    else:
        question = sys.argv[1]
        chunks = retrieve(question, top_k=5)
        if not chunks:
            print("No chunks found — did you run embed_store.py first?")
        else:
            answer = generate_answer(question, chunks)
            print(f"Q: {question}\n")
            print(f"A: {answer}")

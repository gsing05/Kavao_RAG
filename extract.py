"""
Step 1: Turn a PDF into text, split into rough sections.

Research papers usually have headers like "Abstract", "Introduction",
"Methods", "Results", "Discussion", "Conclusion", "References". This
module pulls raw text out of a PDF and does a best-effort split on
those headers so the chunker (chunk.py) has natural boundaries to work
with. PDF extraction is messy — headers don't always come through
cleanly formatted — so this is intentionally forgiving: anything it
can't confidently match to a header just stays in a "body" section
rather than getting lost.
"""

import re
from pathlib import Path
from pypdf import PdfReader

# Common section header names in research papers, in a typical order.
# Matched case-insensitively, allowing for numbering like "1. Introduction".
SECTION_HEADERS = [
    "abstract",
    "introduction",
    "background",
    "literature review",
    "methods",
    "methodology",
    "materials and methods",
    "results",
    "results and discussion",
    "discussion",
    "conclusion",
    "conclusions",
    "acknowledgments",
    "acknowledgements",
    "references",
]

# Matches a line that is JUST a header (optionally numbered), e.g.:
#   "3. Methods"  /  "METHODS"  /  "Discussion"
_HEADER_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)]?\s*)?(" + "|".join(SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE,
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull all text out of a PDF, page by page, joined with newlines."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


# Papers with literature-summary tables often have a "References" column
# (per-row citation numbers). When the PDF is flattened to plain text,
# that column header becomes a standalone line reading just "References",
# which otherwise looks identical to a real section header. A genuine
# References SECTION is followed shortly by an actual numbered citation
# list (e.g. "1. Smith, J. et al...") -- a table column header is not.
_CITATION_START_PATTERN = re.compile(r"^\s*\[?\d+[\.\]]\s+\S")


def _looks_like_real_references_start(lines: list, header_line_index: int,
                                       lookahead: int = 5) -> bool:
    """Peek at the next few non-empty lines after a 'references' header
    match to see if they look like a real numbered citation list."""
    checked = 0
    for line in lines[header_line_index + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _CITATION_START_PATTERN.match(stripped):
            return True
        checked += 1
        if checked >= lookahead:
            break
    return False


def split_into_sections(raw_text: str) -> dict:
    """
    Best-effort split of raw paper text into {section_name: text}.

    If no headers are detected at all, the whole paper is returned
    under a single "body" key — the chunker's fixed-size fallback
    will still handle it fine, just without section-level metadata.
    """
    lines = raw_text.split("\n")
    sections = {}
    current_section = "body"
    current_lines = []

    for i, line in enumerate(lines):
        match = _HEADER_PATTERN.match(line.strip())
        if match and match.group(1).lower() == "references" and \
                not _looks_like_real_references_start(lines, i):
            # Looks like a table's "References" column header, not a real
            # section boundary -- treat this line as ordinary content.
            current_lines.append(line)
            continue
        if match:
            # Save whatever we've collected under the previous header
            if current_lines:
                sections.setdefault(current_section, "")
                sections[current_section] += "\n".join(current_lines) + "\n"
            current_section = match.group(1).lower()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        sections.setdefault(current_section, "")
        sections[current_section] += "\n".join(current_lines) + "\n"

    return sections


def process_paper(pdf_path: str) -> dict:
    """
    Full step-1 pipeline for one paper: extract text, split into sections.
    Returns a dict with the source filename and its sections.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(raw_text)
    return {
        "source": Path(pdf_path).name,
        "sections": sections,
    }


if __name__ == "__main__":
    # Quick manual check: run `python src/extract.py papers/somefile.pdf`
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/extract.py <path_to_pdf>")
    else:
        result = process_paper(sys.argv[1])
        print(f"Source: {result['source']}")
        print(f"Sections found: {list(result['sections'].keys())}")
        for name, text in result["sections"].items():
            preview = text.strip().replace("\n", " ")[:150]
            print(f"\n--- {name} ({len(text)} chars) ---\n{preview}...")

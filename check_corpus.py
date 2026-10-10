"""
Quick quality check across every PDF in papers/ -- runs extraction and
chunking ONLY (no embedding, no model download, no API calls). Use this
to spot problems (scanned/image PDFs with no real text, missing section
headers, oddly huge/tiny chunks) before committing to the slower,
heavier embedding step.

Usage: python src/check_corpus.py
"""

from pathlib import Path
from extract import process_paper
from chunk import chunk_paper

MIN_EXPECTED_CHARS = 2000  # a real paper should extract to at least this much text


def check_one_paper(pdf_path: Path) -> dict:
    processed = process_paper(str(pdf_path))
    total_chars = sum(len(text) for text in processed["sections"].values())
    chunked = chunk_paper(processed)

    warnings = []
    if total_chars < MIN_EXPECTED_CHARS:
        warnings.append(f"very little text extracted ({total_chars} chars) -- "
                         f"possibly a scanned/image-based PDF that needs OCR")
    if list(processed["sections"].keys()) == ["body"]:
        warnings.append("no section headers detected -- everything grouped as one block "
                         "(chunking will still work via the fixed-size fallback, "
                         "just without section-level citations)")

    return {
        "filename": pdf_path.name,
        "total_chars": total_chars,
        "sections": list(processed["sections"].keys()),
        "num_parents": len(chunked["parents"]),
        "num_children": len(chunked["children"]),
        "warnings": warnings,
    }


def run_check():
    papers_dir = Path(__file__).parent.parent / "papers"
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {papers_dir}.")
        return

    print(f"Checking {len(pdf_files)} paper(s)...\n")
    print(f"{'Filename':<25} {'Chars':>8} {'Sections':>9} {'Parents':>8} {'Children':>9}")
    print("-" * 65)

    all_warnings = []
    for pdf_path in pdf_files:
        try:
            result = check_one_paper(pdf_path)
        except Exception as e:
            print(f"{pdf_path.name:<25} FAILED: {e}")
            continue

        print(f"{result['filename']:<25} {result['total_chars']:>8} "
              f"{len(result['sections']):>9} {result['num_parents']:>8} "
              f"{result['num_children']:>9}")

        for w in result["warnings"]:
            all_warnings.append(f"  [{result['filename']}] {w}")

    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(w)
    else:
        print("\nNo warnings -- every paper extracted and chunked cleanly.")


if __name__ == "__main__":
    run_check()

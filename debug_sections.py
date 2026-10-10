"""
Diagnostic tool: shows the raw text immediately before and after each
place a section header gets matched in a PDF. Use this to see WHY a
header landed where it did -- e.g. to check whether "References" is
genuinely marking the bibliography, or whether column-reordering during
PDF extraction has placed that header next to unrelated body text.

Usage: python src/debug_sections.py papers/paper006.pdf references
       (second argument is optional -- omit it to see ALL header matches)
"""

import sys
import re
from extract import extract_text_from_pdf, _HEADER_PATTERN


def debug_headers(pdf_path: str, target_section: str = None, context_chars: int = 400):
    raw_text = extract_text_from_pdf(pdf_path)
    lines = raw_text.split("\n")

    # Rebuild character offsets per line so we can show surrounding context
    offset = 0
    matches = []
    for line in lines:
        match = _HEADER_PATTERN.match(line.strip())
        if match:
            section_name = match.group(1).lower()
            if target_section is None or section_name == target_section.lower():
                matches.append((section_name, offset))
        offset += len(line) + 1  # +1 for the newline we split on

    if not matches:
        print(f"No header matches found for '{target_section}' in {pdf_path}")
        return

    print(f"Found {len(matches)} match(es) for "
          f"{'all headers' if target_section is None else repr(target_section)} "
          f"in {pdf_path}\n")

    for section_name, pos in matches:
        start = max(0, pos - context_chars)
        end = min(len(raw_text), pos + context_chars)
        print("=" * 70)
        print(f"HEADER MATCHED: '{section_name}' at character position {pos}")
        print("=" * 70)
        print("...BEFORE:")
        print(raw_text[start:pos].replace("\n", " ⏎ "))
        print("\n>>> HEADER LINE HERE <<<\n")
        print("AFTER...:")
        print(raw_text[pos:end].replace("\n", " ⏎ "))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/debug_sections.py <path_to_pdf> [section_name]")
    else:
        pdf_path = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else None
        debug_headers(pdf_path, target)

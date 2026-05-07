"""
ingest.py — PDF ingestion pipeline.

Extracts text from a PDF using pymupdf (fitz), splits into section chunks
by detecting standard academic section headers, and returns structured dicts.

Entry point:
  ingest_pdf(pdf_path: str) -> list[dict]

Each chunk:
  {
    "section": str,        # detected section name, or "PREAMBLE" before first header
    "text": str,           # full text of the section
    "page_start": int,     # 1-indexed page where section begins
    "page_end": int,       # 1-indexed page where section ends
    "source_file": str,    # basename of the PDF
    "has_scanned_pages": bool,  # True if any page in range had empty text
  }
"""

import re
import sys
from pathlib import Path

import fitz  # pymupdf

# Headers to detect, in order of priority for matching.
# Match as full line, optional leading number/dot, case-insensitive.
_SECTION_NAMES = [
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "patients and methods",
    "study design",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "acknowledgements",
    "acknowledgments",
    "funding",
    "supplementary",
]

_HEADER_RE = re.compile(
    r"^\s*(?:\d+[\.\d]*\s+)?(" + "|".join(re.escape(s) for s in _SECTION_NAMES) + r")\s*[:\.]?\s*$",
    re.IGNORECASE,
)


def _extract_pages(pdf_path: str) -> list[dict]:
    """Return per-page text with metadata. Empty text pages flagged as scanned."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        page_width = page.rect.width
        midpoint = page_width / 2

        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

        if text_blocks:
            left = [b for b in text_blocks if (b[0] + b[2]) / 2 < midpoint]
            right = [b for b in text_blocks if (b[0] + b[2]) / 2 >= midpoint]
            is_two_column = len(left) >= 2 and len(right) >= 2

            if is_two_column:
                left.sort(key=lambda b: b[1])
                right.sort(key=lambda b: b[1])
                text = "\n".join(b[4] for b in left) + "\n" + "\n".join(b[4] for b in right)
            else:
                text_blocks.sort(key=lambda b: b[1])
                text = "\n".join(b[4] for b in text_blocks)
        else:
            text = ""
            is_two_column = False

        try:
            has_table = bool(page.find_tables().tables)
        except Exception:
            has_table = False

        pages.append({
            "page_num": i + 1,
            "text": text,
            "is_scanned": len(text.strip()) == 0,
            "two_column": is_two_column,
            "has_table": has_table,
        })
    doc.close()
    return pages


def _detect_sections(pages: list[dict], source_file: str) -> list[dict]:
    """
    Walk pages line-by-line, detect section header transitions.
    Returns list of section chunks with accumulated text and page ranges.
    """
    chunks: list[dict] = []

    current_section = "PREAMBLE"
    current_lines: list[str] = []
    current_page_start = 1
    current_has_scanned = False
    current_two_column = False
    current_has_table = False

    def _flush(section, lines, page_start, page_end, has_scanned, two_column, has_table):
        text = "\n".join(lines).strip()
        if text or has_scanned:
            chunks.append({
                "section": section,
                "text": text,
                "page_start": page_start,
                "page_end": page_end,
                "source_file": source_file,
                "has_scanned_pages": has_scanned,
                "two_column": two_column,
                "has_table": has_table,
            })

    for page in pages:
        page_num = page["page_num"]
        if page["is_scanned"]:
            current_has_scanned = True
            continue

        current_two_column = current_two_column or page.get("two_column", False)
        current_has_table = current_has_table or page.get("has_table", False)

        for line in page["text"].splitlines():
            m = _HEADER_RE.match(line)
            if m:
                # Flush previous section
                _flush(
                    current_section,
                    current_lines,
                    current_page_start,
                    page_num,
                    current_has_scanned,
                    current_two_column,
                    current_has_table,
                )
                # Start new section
                current_section = m.group(1).strip().title()
                current_lines = []
                current_page_start = page_num
                current_has_scanned = page["is_scanned"]
                current_two_column = page.get("two_column", False)
                current_has_table = page.get("has_table", False)
            else:
                current_lines.append(line)

    # Flush final section
    _flush(
        current_section,
        current_lines,
        current_page_start,
        pages[-1]["page_num"] if pages else 1,
        current_has_scanned,
        current_two_column,
        current_has_table,
    )

    return chunks


def ingest_pdf(pdf_path: str) -> list[dict]:
    """
    Main entry point. Accepts path to a PDF, returns list of section chunks.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = _extract_pages(str(path))
    chunks = _detect_sections(pages, source_file=path.name)
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path/to/paper.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chunks = ingest_pdf(pdf_path)

    print(f"\nFound {len(chunks)} section(s) in '{Path(pdf_path).name}':\n")
    for i, chunk in enumerate(chunks):
        scanned_note = " [SCANNED PAGES]" if chunk["has_scanned_pages"] else ""
        preview = chunk["text"][:120].replace("\n", " ")
        print(
            f"[{i+1}] {chunk['section']:<25} "
            f"pp.{chunk['page_start']}-{chunk['page_end']}"
            f"{scanned_note}"
        )
        print(f"     {preview!r}")
        print()


if __name__ == "__main__":
    main()

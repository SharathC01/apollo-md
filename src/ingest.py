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
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,  # 1-indexed
            "text": text,
            "is_scanned": len(text.strip()) == 0,
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

    def _flush(section, lines, page_start, page_end, has_scanned):
        text = "\n".join(lines).strip()
        if text or has_scanned:
            chunks.append({
                "section": section,
                "text": text,
                "page_start": page_start,
                "page_end": page_end,
                "source_file": source_file,
                "has_scanned_pages": has_scanned,
            })

    for page in pages:
        page_num = page["page_num"]
        if page["is_scanned"]:
            current_has_scanned = True
            continue

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
                )
                # Start new section
                current_section = m.group(1).strip().title()
                current_lines = []
                current_page_start = page_num
                current_has_scanned = page["is_scanned"]
            else:
                current_lines.append(line)

    # Flush final section
    _flush(
        current_section,
        current_lines,
        current_page_start,
        pages[-1]["page_num"] if pages else 1,
        current_has_scanned,
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

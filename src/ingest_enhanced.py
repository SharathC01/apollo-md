"""
ingest_enhanced.py — Enhanced PDF ingestion with two-column layout and table extraction.

Functions:
  extract_page_text(page) -> tuple[str, bool]
  extract_tables(page) -> str
  parse_pdf_enhanced(path: str) -> list[dict]
  load_all_papers_enhanced(papers_dir="data/raw") -> list[dict]

Each chunk dict: {text, page, source_file, has_table, two_column}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
import pandas as pd


def extract_page_text(page) -> tuple[str, bool]:
    """
    Extract text from a page using block-level position info.

    Detects two-column layout by checking whether text blocks cluster on
    both sides of the page midpoint. If two-column, reads left column
    top-to-bottom then right column top-to-bottom.

    Returns (combined_text, is_two_column).
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    page_width = page.rect.width
    midpoint = page_width / 2

    # Keep only text blocks (block_type == 0)
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    if not text_blocks:
        return "", False

    left = [b for b in text_blocks if (b[0] + b[2]) / 2 < midpoint]
    right = [b for b in text_blocks if (b[0] + b[2]) / 2 >= midpoint]

    # Need meaningful content on both sides to be considered two-column
    is_two_column = len(left) >= 2 and len(right) >= 2

    if is_two_column:
        left.sort(key=lambda b: b[1])
        right.sort(key=lambda b: b[1])
        combined = "\n".join(b[4] for b in left) + "\n" + "\n".join(b[4] for b in right)
    else:
        text_blocks.sort(key=lambda b: b[1])
        combined = "\n".join(b[4] for b in text_blocks)

    return combined.strip(), is_two_column


def extract_tables(page) -> str:
    """
    Extract structured tables from a page using pymupdf's find_tables().

    Converts each table to a pandas DataFrame then to a string.
    Prefixes each table with 'TABLE N:'.
    Returns empty string if no tables found.
    """
    try:
        table_finder = page.find_tables()
    except Exception:
        return ""

    if not table_finder.tables:
        return ""

    parts = []
    for i, table in enumerate(table_finder.tables, start=1):
        try:
            data = table.extract()
            df = pd.DataFrame(data)
            parts.append(f"TABLE {i}:\n{df.to_string(index=False, header=False)}")
        except Exception:
            continue

    return "\n\n".join(parts)


def parse_pdf_enhanced(path: str) -> list[dict]:
    """
    Parse a PDF into per-page chunks combining body text and tables.

    Each chunk dict contains:
      text        — full page text including any table representations
      page        — 1-indexed page number
      source_file — basename of the PDF
      has_table   — True if at least one table was found on this page
      two_column  — True if two-column layout was detected
    """
    doc = fitz.open(path)
    source_file = Path(path).name
    chunks = []

    for page in doc:
        text, is_two_column = extract_page_text(page)
        table_text = extract_tables(page)
        has_table = bool(table_text)
        full_text = (text + "\n\n" + table_text).strip() if has_table else text

        chunks.append({
            "text": full_text,
            "page": page.number + 1,
            "source_file": source_file,
            "has_table": has_table,
            "two_column": is_two_column,
        })

    doc.close()
    return chunks


def load_all_papers_enhanced(papers_dir: str = "data/raw") -> list[dict]:
    """
    Load all PDFs from papers_dir and return all page chunks across all papers.

    Prints per-paper stats: pages, table pages, two-column pages.
    Skips non-PDF files and logs errors per paper without crashing.
    """
    papers_path = Path(__file__).parent.parent / papers_dir
    pdfs = sorted(papers_path.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {papers_path}")
        return []

    all_chunks: list[dict] = []

    for pdf in pdfs:
        try:
            chunks = parse_pdf_enhanced(str(pdf))
            table_pages = sum(1 for c in chunks if c["has_table"])
            two_col_pages = sum(1 for c in chunks if c["two_column"])
            print(
                f"  {pdf.name}: {len(chunks)} pages, "
                f"{table_pages} table pages, {two_col_pages} two-column pages"
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"  ERROR {pdf.name}: {exc}")

    return all_chunks


if __name__ == "__main__":
    import sys

    papers_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    print(f"Loading papers from {papers_dir}...\n")
    chunks = load_all_papers_enhanced(papers_dir)
    print(f"\nTotal chunks: {len(chunks)}")
    table_total = sum(1 for c in chunks if c["has_table"])
    two_col_total = sum(1 for c in chunks if c["two_column"])
    print(f"Pages with tables: {table_total}")
    print(f"Two-column pages:  {two_col_total}")

"""
run_batch.py — Batch extraction runner.

Loops over all PDFs in data/raw/, skips those already extracted,
calls extract_pdf() for each, and prints a final summary.

Usage:
  python run_batch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.extractor import extract_pdf

RAW_DIR = Path(__file__).parent / "data" / "raw"
EXTRACTED_DIR = Path(__file__).parent / "data" / "extracted"


def main():
    pdfs = sorted(p for p in RAW_DIR.iterdir() if p.suffix.lower() == ".pdf")
    total = len(pdfs)

    if total == 0:
        print(f"No PDFs found in {RAW_DIR}")
        sys.exit(0)

    succeeded = []
    failed = []
    skipped = []

    for i, pdf in enumerate(pdfs, start=1):
        label = f"{i}/{total}: {pdf.name}"
        out_json = EXTRACTED_DIR / f"{pdf.stem}.json"

        if out_json.exists():
            print(f"Skipping  {label}  (already extracted)")
            skipped.append(pdf.name)
            continue

        print(f"Processing {label} ...", end=" ", flush=True)
        try:
            _, out_path = extract_pdf(str(pdf))
            print(f"OK → {out_path.name}")
            succeeded.append(pdf.name)
        except Exception as exc:
            print(f"FAILED")
            print(f"  Error: {exc}")
            failed.append((pdf.name, str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed, {len(skipped)} skipped.")
    if failed:
        print("\nFailed files:")
        for name, err in failed:
            short_err = err.splitlines()[0][:120]
            print(f"  ✗ {name}: {short_err}")
    print("=" * 60)


if __name__ == "__main__":
    main()

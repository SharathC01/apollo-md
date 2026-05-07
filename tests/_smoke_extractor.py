"""Smoke test for extractor helpers — no API calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import _filter_chunks, _build_user_message, _parse_response
from src.ingest import ingest_pdf

chunks = ingest_pdf("papers/cureus-0014-00000030887.pdf")
filtered = _filter_chunks(chunks)
print(f"Total chunks: {len(chunks)}")
print(f"Filtered (Methods+Results): {len(filtered)}")
for c in filtered:
    print(f"  {c['section']} pp.{c['page_start']}-{c['page_end']} ({len(c['text'])} chars)")

# JSON parse — bare
raw = '{"title": "Test", "year": 2024, "mortality_predictors": []}'
parsed = _parse_response(raw, "test.pdf")
print(f"\nParse bare JSON: title={parsed['title']}")

# JSON parse — fenced
fenced = "```json\n{\"title\": \"Fenced\"}\n```"
parsed2 = _parse_response(fenced, "test.pdf")
print(f"Parse fenced JSON: title={parsed2['title']}")

# User message structure
msg = _build_user_message(filtered, "cureus-0014-00000030887.pdf")
print(f"\nUser message: {len(msg)} chars, {len(filtered)} sections embedded")
print("OK — all helpers pass")

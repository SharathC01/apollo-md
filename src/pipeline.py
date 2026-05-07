"""
pipeline.py — Unified query pipeline connecting all components.

Step 1: Load validated records from data/extracted/
Step 2: Filter by keyword query
Step 3: If >= 3 matches, build evidence table
Step 4: If < 3 matches, fall back to RAG over raw PDF chunks
Step 5: Return appropriate DataFrame based on use_case

Functions:
  run_pipeline(query, use_case, top_k) -> pd.DataFrame
  get_source_quote(study_id, predictor) -> str
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.validate_enhanced import validate_all
from src.table_builder import build_evidence_table, build_ranked_predictors, filter_by_query
from src.ingest_enhanced import load_all_papers_enhanced
import src.retrieve as _retrieve

# Cache validated records for the lifetime of the process
_validated_records: list[dict] | None = None


def _load_records() -> list[dict]:
    """Return cached validated records, loading on first call."""
    global _validated_records
    if _validated_records is None:
        _validated_records = validate_all()
    return _validated_records


def run_pipeline(
    query: str,
    use_case: str = "mortality",
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Execute the full query pipeline and return a structured evidence DataFrame.

    use_case options:
      "mortality"  -> build_evidence_table() on filtered records
      "biomarker"  -> build_ranked_predictors() (AUC-ranked) on filtered records
      "phenotype"  -> build_evidence_table() on filtered records

    Falls back to RAG (semantic search over raw PDF chunks) when fewer than
    3 keyword matches are found. RAG fallback returns a DataFrame with columns:
    source_file, page, score, text_preview.
    """
    # Step 1: load validated records
    records = _load_records()

    # Step 2: keyword filter
    filtered = filter_by_query(records, query)

    # Step 3: enough keyword matches — use structured path
    if len(filtered) >= 3:
        if use_case == "biomarker":
            return build_ranked_predictors(filtered)
        else:
            return build_evidence_table(filtered)

    # Step 4: RAG fallback
    print(
        f"Only {len(filtered)} keyword match(es) for '{query}' — "
        f"falling back to semantic search over raw PDF chunks..."
    )
    chunks = load_all_papers_enhanced()
    _, embeddings = _retrieve.embed_chunks(chunks)
    top_chunks = _retrieve.retrieve(query, chunks, embeddings, top_k=top_k)

    return pd.DataFrame([
        {
            "source_file": c.get("source_file"),
            "page": c.get("page"),
            "score": round(c.get("score", 0.0), 4),
            "text_preview": str(c.get("text") or "")[:200],
        }
        for c in top_chunks
    ])


def get_source_quote(study_id: str, predictor: str) -> str:
    """
    Look up the verbatim source quote for a given study + predictor combination.

    Matching is case-insensitive and partial (substring). Returns 'not reported'
    if no matching record is found.
    """
    records = _load_records()
    sid_lower = study_id.lower()
    pred_lower = predictor.lower()

    for r in records:
        study_match = sid_lower in str(r.get("study_id") or "").lower()
        pred_match = pred_lower in str(r.get("predictor") or "").lower()
        if study_match and pred_match:
            return r.get("source_quote") or "not reported"

    return "not reported"


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "SOFA score and mortality"
    use_case = sys.argv[2] if len(sys.argv) > 2 else "mortality"

    print(f"\nRunning pipeline: query='{query}' use_case='{use_case}'\n")
    df = run_pipeline(query, use_case=use_case)

    if "Study" in df.columns:
        print(df[["Study", "Predictor", "AUC", "Verified", "Confidence"]].head(10).to_string())
    else:
        print(df.head(10).to_string())

    print(f"\nPipeline working: {len(df)} records found")

"""
pipeline.py — Unified query pipeline connecting all components.

Step 1: Load validated records from data/extracted/
Step 2: Parse query → extract predictor/outcome intent
Step 3: If predictor found → filter corpus → structured table
Step 4: If no predictor or no corpus hit → RAG semantic search
Step 5: Return (df, summary)

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
from src.query_parser import parse_query
from src.summarizer import summarize_evidence, summarize_phenotype
import src.retrieve as _retrieve

CORE_PAPERS = [
    "Seymour_2016", "Baloch_2022", "Besen_2016", "Wang_2020",
    "Li_2020", "Roh_2019", "Zhang_2019", "Koozi_2023",
    "Todi_2024", "Varga_2024"
]

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
    debug: bool = False,
    core_only: bool = True,
) -> tuple[pd.DataFrame, str]:
    """
    Execute the full query pipeline.

    Returns (df, summary) where:
      df      — structured evidence DataFrame
      summary — 2-3 sentence LLM summary (empty string for RAG path)

    use_case options:
      "mortality"  -> build_evidence_table() on filtered records
      "biomarker"  -> build_ranked_predictors() (AUC-ranked) on filtered records
      "phenotype"  -> build_evidence_table() on filtered records

    Routing:
      predictor identified + corpus hit  → structured path (result_type="structured")
      predictor identified + no hit      → RAG fallback   (result_type="semantic")
      no predictor identified            → RAG fallback   (result_type="semantic")
    """
    # Step 1: parse query intent
    parsed = parse_query(query)
    print(
        f"Query parsed: predictor={parsed.get('predictor')} "
        f"outcome={parsed.get('outcome')} "
        f"keywords={parsed.get('keywords')}"
    )

    # Step 2: load validated records
    all_records = _load_records()
    if core_only:
        all_records = [r for r in all_records if r.get("study_id", "").split()[0] in CORE_PAPERS]

    # Step 3: predictor-anchored structured path
    if parsed.get("predictor"):
        filtered = filter_by_query(all_records, parsed_query=parsed, debug=debug)

        if debug and filtered:
            print("\n[DEBUG] Filter matches:")
            hdr = f"{'Study':<20} {'Predictor':<35} {'Outcome':<30} {'Match Reason'}"
            print(hdr)
            print("-" * len(hdr))
            for r in filtered:
                print(
                    f"{str(r.get('study_id') or ''):<20} "
                    f"{str(r.get('predictor') or '')[:34]:<35} "
                    f"{str(r.get('outcome') or '')[:29]:<30} "
                    f"{r.get('match_reason', '')}"
                )
            print()

        if len(filtered) > 0:
            if use_case == "biomarker":
                df = build_ranked_predictors(filtered)
            else:
                df = build_evidence_table(filtered)
            df["result_type"] = "structured"
            if use_case == "phenotype":
                summary = summarize_phenotype(df, parsed)
            else:
                summary = summarize_evidence(df, parsed)
            return df, summary

        print(f"Predictor '{parsed['predictor']}' not found in corpus — falling back to RAG...")
    else:
        print(f"No predictor identified in query — falling back to RAG...")

    # Step 4: RAG fallback — semantic search over structured evidence records
    rag_records, embeddings = _retrieve.build_evidence_index()
    top_records = _retrieve.retrieve(query, rag_records, embeddings, top_k=top_k)
    df = build_evidence_table(top_records)
    df["result_type"] = "semantic"
    if use_case == "phenotype":
        summary = summarize_phenotype(df, parsed)
    else:
        summary = ""
    return df, summary


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
    df, summary = run_pipeline(query, use_case=use_case)

    if summary:
        print(f"Summary:\n{summary}\n")

    print(df[["Study", "Predictor", "AUC", "Verified", "Confidence", "result_type"]].head(10).to_string())

    print(f"\nPipeline working: {len(df)} records found")

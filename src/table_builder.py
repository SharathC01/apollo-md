"""
table_builder.py — Evidence table construction and ranking.

Converts flat validated records (from validate_enhanced) into display-ready
pandas DataFrames, with sorting, AUC extraction, and query filtering.

Functions:
  build_evidence_table(records) -> pd.DataFrame
  build_ranked_predictors(records) -> pd.DataFrame
  filter_by_query(records, parsed_query, debug=False) -> list[dict]
  save_table(df, path) -> None
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2, "unverified": 3}


def _nr(val) -> str:
    """Return 'not reported' for any falsy or null-like value."""
    if val is None or val == "":
        return "not reported"
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return "not reported"
    return val


def build_evidence_table(records: list[dict]) -> pd.DataFrame:
    """
    Build a display-ready evidence table from validated records.

    Columns: Study, Country, N, Predictor, Outcome, Timing, Method,
             Effect Size, AUC, Cutoff, Adjustment, Verified, Confidence,
             Source Quote (first 100 chars), Page, File.

    Sorted by confidence tier: high → medium → low → unverified.
    Null/empty values replaced with 'not reported'.
    """
    rows = []
    for r in records:
        rows.append({
            "Study": _nr(r.get("study_id")),
            "Country": _nr(r.get("country")),
            "N": _nr(r.get("sample_size")),
            "Predictor": _nr(r.get("predictor")),
            "Outcome": _nr(r.get("outcome")),
            "Timing": _nr(r.get("timing")),
            "Method": _nr(r.get("method")),
            "Effect Size": _nr(r.get("effect_size")),
            "AUC": _nr(r.get("auc")),
            "Cutoff": _nr(r.get("cutoff")),
            "Adjustment": _nr(r.get("adjustment")),
            "Verified": "✓" if r.get("verified") else "✗",
            "Confidence": _nr(r.get("confidence")),
            "Source Quote": str(r.get("source_quote") or "")[:100],
            "Page": _nr(r.get("page")),
            "File": _nr(r.get("source_file")),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_sort"] = df["Confidence"].map(lambda c: _CONFIDENCE_ORDER.get(c, 99))
        df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def _extract_auc_numeric(auc_str) -> float | None:
    """Parse first float from strings like '0.74 (95% CI, 0.73-0.76)'."""
    if not auc_str or auc_str == "not reported":
        return None
    m = re.search(r"(\d+\.\d+)", str(auc_str))
    return float(m.group(1)) if m else None


def build_ranked_predictors(records: list[dict]) -> pd.DataFrame:
    """
    Build a predictor ranking table sorted by AUC descending.

    Filters to records with parseable AUC values only.
    Columns: Predictor, AUC (numeric), AUC (full), Study, Outcome, Method, Verified.
    """
    rows = []
    for r in records:
        auc_str = r.get("auc") or ""
        auc_num = _extract_auc_numeric(auc_str)
        if auc_num is None:
            continue
        rows.append({
            "Predictor": _nr(r.get("predictor")),
            "AUC": auc_num,
            "AUC (full)": _nr(auc_str),
            "Study": _nr(r.get("study_id")),
            "Outcome": _nr(r.get("outcome")),
            "Method": _nr(r.get("method")),
            "Verified": "✓" if r.get("verified") else "✗",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("AUC", ascending=False).reset_index(drop=True)
    return df


def _first_match(text: str, words: list[str]) -> str | None:
    """Return first word that appears in text, or None."""
    t = text.lower()
    for w in words:
        if w in t:
            return w
    return None


def filter_by_query(
    records: list[dict],
    parsed_query: dict,
    debug: bool = False,
) -> list[dict]:
    """
    Filter records by parsed predictor term only. No keyword fallback.

    Matching order:
    1. Full phrase in record["predictor"]
    2. If 0 results: individual meaningful words (len >= 4) in record["predictor"]
    3. If still 0: phrase + words in record["source_quote"]

    Returns empty list if parsed_query has no predictor or nothing matches.
    When debug=True, each returned record gets a "match_reason" field.
    """
    def _tag(r: dict, reason: str) -> dict:
        if debug:
            out = dict(r)
            out["match_reason"] = reason
            return out
        return r

    pred_term = (parsed_query.get("predictor") or "").lower().strip()
    if not pred_term:
        return []

    # Step 1: full phrase in predictor field
    results = []
    for r in records:
        if pred_term in str(r.get("predictor") or "").lower():
            results.append(_tag(r, f"matched_predictor: {pred_term}"))
    if results:
        return results

    # Step 2: individual meaningful words in predictor field
    pred_words = [w for w in pred_term.split() if len(w) >= 4]
    if pred_words:
        for r in records:
            hit = _first_match(str(r.get("predictor") or ""), pred_words)
            if hit:
                results.append(_tag(r, f"matched_predictor: {hit}"))
        if results:
            return results

    # Step 3: phrase + words in source_quote
    search_terms = ([pred_term] + pred_words) if pred_words else [pred_term]
    for r in records:
        hit = _first_match(str(r.get("source_quote") or ""), search_terms)
        if hit:
            results.append(_tag(r, f"matched_source_quote: {hit}"))
    return results


def save_table(df: pd.DataFrame, path: str = "results/evidence_table.csv") -> None:
    """
    Save DataFrame to CSV, creating parent directories as needed.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved → {out} ({len(df)} rows)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.validate_enhanced import validate_all

    records = validate_all()
    print(f"\nBuilding evidence table from {len(records)} records...")
    df = build_evidence_table(records)
    print(df[["Study", "Predictor", "AUC", "Verified", "Confidence"]].head(10).to_string())

    print("\nBuilding ranked predictors (AUC only)...")
    ranked = build_ranked_predictors(records)
    print(f"Ranked {len(ranked)} predictors with AUC values.")
    print(ranked.head(10).to_string())

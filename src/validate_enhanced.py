"""
validate_enhanced.py — Machine-verified validation with confidence scoring.

Flattens nested mortality_predictor dicts from extracted JSONs into tabular
records with source_quote verification and tiered confidence labels.

Functions:
  validate_record(predictor, study_meta) -> dict
  validate_study(json_path) -> list[dict]
  validate_all(extracted_dir="data/extracted") -> list[dict]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_record(predictor: dict, study_meta: dict) -> dict:
    """
    Flatten one mortality_predictor dict + parent study metadata into a
    single validated record.

    Confidence tiers:
      high       — has AUC or effect_size, verified quote, raw confidence >= 0.9
      medium     — has AUC or effect_size, verified quote
      low        — verified quote only
      unverified — missing or too-short source quote
    """
    source_quote = predictor.get("source_quote") or ""
    verified = bool(source_quote and len(source_quote) > 10)

    auc = predictor.get("auc")
    effect_size = predictor.get("effect_size")
    has_evidence = bool(auc or effect_size)
    raw_confidence = float(predictor.get("confidence") or 0)

    if has_evidence and verified and raw_confidence >= 0.9:
        confidence_tier = "high"
    elif has_evidence and verified:
        confidence_tier = "medium"
    elif verified:
        confidence_tier = "low"
    else:
        confidence_tier = "unverified"

    meta = study_meta.get("_meta") or {}
    sample_size_obj = study_meta.get("sample_size") or {}

    return {
        "study_id": meta.get("paper_id") or study_meta.get("title") or "unknown",
        "source_file": meta.get("source_file") or "unknown",
        "country": study_meta.get("country") or "not reported",
        "sample_size": sample_size_obj.get("value") if sample_size_obj else None,
        "predictor": predictor.get("predictor_variable") or "not reported",
        "outcome": predictor.get("outcome_definition") or "not reported",
        "timing": predictor.get("timing") or "not reported",
        "method": predictor.get("statistical_method") or "not reported",
        "effect_size": effect_size or "not reported",
        "auc": auc or "not reported",
        "sensitivity": predictor.get("sensitivity") or "not reported",
        "specificity": predictor.get("specificity") or "not reported",
        "cutoff": predictor.get("cutoff_value") or "not reported",
        "adjustment": predictor.get("confounders_adjusted") or "not reported",
        "source_quote": source_quote or "not reported",
        "page": predictor.get("page"),
        "raw_confidence": raw_confidence,
        "verified": verified,
        "confidence": confidence_tier,
    }


def validate_study(json_path: str) -> list[dict]:
    """
    Load one extracted JSON file and validate all mortality_predictors.

    Returns list of flat validated records (one per predictor entry).
    Returns empty list if mortality_predictors is absent or empty.
    """
    with open(json_path, encoding="utf-8") as f:
        study = json.load(f)

    predictors = study.get("mortality_predictors") or []
    return [validate_record(p, study) for p in predictors]


def validate_all(extracted_dir: str = "data/extracted") -> list[dict]:
    """
    Run validate_study on every JSON file in extracted_dir.

    Prints per-study predictor count and a final total summary.
    Skips files that error without crashing the batch.
    Returns all records as a flat list.
    """
    dir_path = Path(__file__).parent.parent / extracted_dir
    json_files = sorted(dir_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {dir_path}")
        return []

    all_records: list[dict] = []
    high = medium = low = unverified = 0

    for jf in json_files:
        try:
            records = validate_study(str(jf))
            for r in records:
                tier = r["confidence"]
                if tier == "high":
                    high += 1
                elif tier == "medium":
                    medium += 1
                elif tier == "low":
                    low += 1
                else:
                    unverified += 1
            print(f"  {jf.name}: {len(records)} predictor(s)")
            all_records.extend(records)
        except Exception as exc:
            print(f"  ERROR {jf.name}: {exc}")

    print(
        f"\nTotal: {len(all_records)} records from {len(json_files)} studies "
        f"| high={high} medium={medium} low={low} unverified={unverified}"
    )
    return all_records


if __name__ == "__main__":
    import sys

    extracted_dir = sys.argv[1] if len(sys.argv) > 1 else "data/extracted"
    print(f"Validating all studies in {extracted_dir}...\n")
    records = validate_all(extracted_dir)
    if records:
        first = records[0]
        print(f"\nSample record ({first['study_id']}):")
        for k, v in first.items():
            val_str = str(v)[:80] if isinstance(v, str) else str(v)
            print(f"  {k:<20}: {val_str}")

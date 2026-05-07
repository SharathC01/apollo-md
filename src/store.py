"""
store.py — Extracted JSON persistence and querying.

Responsibilities:
  - Save validated SepsisStudy JSON to data/extracted/<paper_id>.json
  - Load all extracted JSONs into a list or pandas DataFrame
  - Provide simple filter/query helpers (by year, design, outcome)
  - No database required — flat JSON files as source of truth

Entry points:
  save(study: SepsisStudy, out_dir: str) -> Path
  load_all(extracted_dir: str) -> list[dict]
  to_dataframe(studies: list[dict]) -> pd.DataFrame
"""

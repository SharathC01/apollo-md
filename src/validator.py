"""
validator.py — Pydantic schema + domain range validation.

Responsibilities:
  - Define SepsisStudy Pydantic model with all clinical fields
  - Enforce field types and Optional/required rules
  - Add domain-range validators (e.g., mortality 0–100%, n > 0)
  - Return validated SepsisStudy instance or raise ValidationError with detail

Schema (planned fields, not exhaustive):
  paper_id, title, year, journal
  study_design, population, n_patients
  intervention, comparator
  primary_outcome, mortality_28d, mortality_90d
  icu_los, hospital_los
  sofa_score_baseline, lactate_baseline
  notes

Entry point:
  validate(raw: dict) -> SepsisStudy
"""

from pydantic import BaseModel

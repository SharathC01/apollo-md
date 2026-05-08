"""
validator.py — Pydantic models for sepsis study extraction schema.

Models:
  MortalityPredictor  — one predictor row from mortality_predictors[]
  SepsisStudy         — top-level extracted study record

Entry point:
  validate_study(raw: dict) -> SepsisStudy
"""

from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class MortalityPredictor(BaseModel):
    predictor_variable: str
    outcome_definition: str
    timing: Optional[str] = None
    statistical_method: str
    association_type: Optional[Literal["modelled", "descriptive"]] = None
    effect_size: Optional[str] = None
    auc: Optional[str] = None
    sensitivity: Optional[str] = None
    specificity: Optional[str] = None
    cutoff_value: Optional[str] = None
    survivors_value: Optional[str] = None
    death_value: Optional[str] = None
    p_value: Optional[str] = None
    confounders_adjusted: Optional[str] = None
    model_context: Optional[str] = None
    source_type: Optional[str] = None
    source_quote: str
    page: Optional[int] = None
    confidence: float

    # New fields — all optional so existing JSONs deserialize without error
    significant: Optional[bool] = None
    # True = significant, False = explicitly non-significant, None = not reported

    comparison_context: Optional[str] = None
    # For head-to-head AUC comparisons, e.g. "Compared to qSOFA AUC 0.65 in same cohort"

    population_age_group: Optional[Literal["adult", "pediatric"]] = None
    # "pediatric" if study population is <18 years or explicitly pediatric

    population_statistic: Optional[str] = None
    # Bare outcome rate with no predictor, e.g. "46% ICU mortality in septic shock"

    effect_size_type: Optional[str] = None
    # Allowed values: "OR", "HR", "RR", "NRI", "IDI", "other"

    model_config = {"extra": "allow"}


class _NumericField(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    source_quote: Optional[str] = None
    page: Optional[int] = None
    confidence: Optional[float] = None

    model_config = {"extra": "allow"}


class SepsisStudy(BaseModel):
    title: Optional[str] = None
    year: Optional[int] = None
    study_design: Optional[str] = None
    country: Optional[str] = None
    sample_size: Optional[_NumericField] = None
    patient_age_mean: Optional[_NumericField] = None
    sofa_score_baseline: Optional[_NumericField] = None
    sofa_score_peak: Optional[_NumericField] = None
    antibiotic_timing_hours: Optional[_NumericField] = None
    icu_mortality_percent: Optional[_NumericField] = None
    mortality_predictors: list[MortalityPredictor] = []

    model_config = {"extra": "allow"}

    @field_validator("mortality_predictors", mode="before")
    @classmethod
    def _coerce_predictors(cls, v):
        return v or []


def validate_study(raw: dict) -> SepsisStudy:
    """Parse and validate a raw extracted study dict. Raises ValidationError on schema violation."""
    return SepsisStudy.model_validate(raw)

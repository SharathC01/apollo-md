"""
synonyms.py — Canonical predictor name normalization.

Maps LLM-extracted free-text predictor names to stable canonical keys
so that downstream filtering and de-duplication work across studies.

Functions:
  normalize_predictor(raw: str) -> str
"""

SYNONYM_MAP: dict[str, list[str]] = {
    "lactate": [
        "serum lactate", "blood lactate", "arterial lactate", "venous lactate",
        "lac", "lactic acid", "hyperlactatemia", "plasma lactate", "serum lactic acid",
    ],
    "procalcitonin": [
        "pct", "serum pct", "plasma procalcitonin", "procalcitonin level",
    ],
    "qsofa": [
        "quick sofa", "quick sequential organ failure assessment", "bedside sepsis criteria",
    ],
    "sofa": [
        "sequential organ failure assessment", "sofa score", "organ failure score",
    ],
    "apache_ii": [
        "apache ii", "apache-ii", "acute physiology and chronic health evaluation ii",
        "apache 2", "apache score",
    ],
    "lods": [
        "logistic organ dysfunction score", "logistic organ dysfunction system",
    ],
    "sirs": [
        "systemic inflammatory response syndrome", "sirs criteria", "sepsis-2 criteria",
    ],
    "altered_mentation": [
        "gcs ≤13", "gcs <13", "gcs <= 13", "confusion", "encephalopathy",
        "altered mental status", "ams", "change in mental status",
        "reduced consciousness", "disorientation",
    ],
    "acute_kidney_injury": [
        "aki", "renal failure", "acute renal failure", "arf",
        "renal dysfunction", "kidney dysfunction", "renal impairment",
    ],
    "creatinine": [
        "serum creatinine", "plasma creatinine", "scr", "creatinine level",
    ],
    "serum_albumin": [
        "albumin", "hypoalbuminemia", "plasma albumin", "albumin level",
    ],
    "rdw": [
        "red cell distribution width", "red blood cell distribution width", "rdw-cv", "rdw-sd",
    ],
    "mcv": [
        "mean corpuscular volume", "mean cell volume",
    ],
    "mechanical_ventilation": [
        "mv", "invasive mechanical ventilation", "intubation",
        "invasive ventilation", "imv", "ventilator use",
    ],
    "vasopressor_use": [
        "vasopressor therapy", "vasopressor requirement", "pressor use",
        "norepinephrine use", "catecholamine use", "pressor therapy",
    ],
    "antibiotic_timing": [
        "time-to-antibiotics", "time to first antibiotic", "antibiotic delay",
        "early antibiotic administration", "time to antimicrobial therapy",
    ],
    "infection_source": [
        "focus of infection", "site of infection", "primary infection site",
        "source of sepsis", "infection focus", "primary source",
    ],
    "hematologic_malignancy": [
        "hemato-oncologic malignancy", "haematological malignancy",
        "blood cancer", "oncologic comorbidity", "hematological neoplasm",
    ],
    "ssc_bundle": [
        "surviving sepsis campaign bundle", "3-hour bundle", "sss compliance",
        "sepsis care bundle", "sepsis resuscitation bundle",
    ],
    "hypotension": [
        "low blood pressure", "systolic hypotension", "sbp < 100",
        "map < 65", "mean arterial pressure < 65",
    ],
    "bacteremia": [
        "bloodstream infection", "positive blood culture", "bacteraemia",
    ],
}

# Reverse lookup: alias_lower → canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SYNONYM_MAP.items():
    _ALIAS_TO_CANONICAL[_canonical.lower()] = _canonical  # canonical maps to itself
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


def normalize_predictor(raw: str) -> str:
    """
    Map a raw predictor string to its canonical name.
    Case-insensitive. Returns input lowercased+stripped if no match found.
    """
    if not raw:
        return ""
    key = raw.strip().lower()
    return _ALIAS_TO_CANONICAL.get(key, key)

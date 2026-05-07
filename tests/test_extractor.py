"""
test_extractor.py — Unit tests for extractor.py and validator.py.

Planned tests:
  - test_extract_returns_expected_fields: mock LLM response → check all required keys present
  - test_validation_passes_valid_study: valid dict → SepsisStudy instance, no exception
  - test_validation_rejects_negative_n: n_patients < 0 → ValidationError
  - test_validation_rejects_mortality_over_100: mortality_28d > 100 → ValidationError
  - test_extract_handles_missing_sections: incomplete chunks → graceful partial extraction
"""

import pytest

Create KNOWN_LIMITATIONS.md in the project root with this content:

# Known Limitations — Apollo MD

## 1. Extraction Schema: Modelled Predictors Only

**Issue:** The extractor captures predictors with modelled effect sizes 
(OR, HR, AUC from regression or ROC analysis) only. Descriptive 
associations — where a biomarker is compared between survival/death 
groups with median values and p-values only (no regression model) — 
are not captured.

**Example:** Cao_2021 Table 4 reports PCT, LAC, SOFA, APACHE II with 
median values for survivors vs non-survivors + p-values. These are 
not extracted as mortality predictors because no OR/HR/AUC is reported.

**Impact:** Queries for biomarkers like PCT return fewer studies than 
actually mention the biomarker in the corpus.

**Fix if time allows:** 
- Add a new schema field: descriptive_associations (list)
- Each item: {variable, survivors_value, death_value, p_value, source_quote, page}
- Update extraction prompt to capture these separately
- Re-extract all 28 papers

## 2. Language Filter Not Implemented

**Issue:** Non-English papers (Kochkin_2021, Kozlov_2022) were not 
explicitly filtered — they failed ingestion due to section detection 
mismatch, not language detection.

**Fix if time allows:**
- Detect language of extracted text (langdetect library)
- Skip non-English papers in run_batch.py with explicit log message
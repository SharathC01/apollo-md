# Apollo MD — Sepsis Evidence Engine

A zero-hallucination medical literature extraction pipeline for sepsis RCTs. Extracts mortality predictors from clinical PDFs, verifies every claim against source text, and surfaces structured evidence through a Streamlit UI.

## What it does

- Ingests sepsis RCT PDFs and extracts structured mortality predictor records via Claude API
- Verifies each extracted quote verbatim against the source PDF text
- Builds a ranked evidence table with AUC, effect sizes, confidence tiers, and GRADE-compatible export
- Answers natural-language queries via keyword-first matching with semantic RAG fallback
- Displays honest uncertainty: unverified quotes, missing confounders, heterogeneous AUCs flagged explicitly

## Current corpus

10 re-extracted papers · 212 records · 75% quote-verified

| Paper | Records |
|-------|---------|
| Besen 2016 | 37 |
| Wang 2020 | 37 |
| Todi 2024 | 25 |
| Seymour 2016 | 23 |
| Varga 2024 | 21 |
| Li 2020 | 24 |
| Koozi 2023 | 13 |
| Roh 2019 | 13 |
| Zhang 2019 | 11 |
| Baloch 2022 | 8 |

## Architecture

```
PDF → ingest.py → extractor.py (Claude API) → data/extracted/*.json
                                                      ↓
                                          validate_enhanced.py
                                          (quote verification + confidence tiers)
                                                      ↓
query → query_parser.py → pipeline.py → table_builder.py → Streamlit UI
                               ↓ (fallback)
                          retrieve.py (RAG / MiniLM embeddings)
```

### Hallucination guards

- **Verbatim quote requirement**: extraction prompt requires `source_quote` copied exactly from PDF text
- **Quote verification**: `verify_quote()` substring-checks every quote against raw PDF text post-extraction
- **Numeric verification**: `verify_numeric()` confirms extracted AUC/OR/HR appears in the source quote
- **Confidence tiering**: `high` requires both quote and numeric verified; `unverified` records displayed with explicit flag
- **Summarizer guardrail**: LLM summary receives only values present in the evidence table; forbidden from using training-data context

### Evidence table columns

`Study · Predictor · Outcome · AUC · Effect Size · Association Type · Verified · Confidence · Page · File`

Association type distinguishes `modelled` (OR/HR/AUC from regression) from `descriptive` (median comparison).

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API key:

```
ANTHROPIC_API_KEY=your_key_here
# OR for OpenRouter:
USE_OPENROUTER=true
OPENROUTER_API_KEY=your_key_here
```

## Run

```bash
streamlit run src/app.py
```

## Re-extract papers

```bash
python run_batch.py
```

## Test

```bash
pytest tests/
```

## Use cases

| Tab | Query type | Output |
|-----|-----------|--------|
| UC1 · Mortality Predictors | `What predicts 28-day mortality in septic shock?` | Evidence table sorted by confidence |
| UC2 · Phenotype Extraction | `What sepsis phenotypes have been identified in ICU patients?` | Phenotype summary + evidence |
| UC3 · Biomarker Ranking | `Which biomarker best predicts sepsis mortality?` | AUC-ranked bar chart + table |

## Known limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

- Quote verification rate: 75% (160/212) — LLM paraphrasing and PDF reflow cause ~25% misses
- Phenotype use case underrepresented in current 10-paper corpus
- Two-column PDF layout causes occasional mid-sentence newlines in extracted text

## Stack

- **Extraction**: Claude API (`claude-sonnet-4-6`) via Anthropic SDK or OpenRouter
- **PDF parsing**: PyMuPDF (`fitz`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Validation**: Pydantic
- **UI**: Streamlit + Plotly
- **Vector search**: NumPy cosine similarity (no external vector DB)

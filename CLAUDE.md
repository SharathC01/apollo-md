# apollo-md

Medical literature extraction pipeline for sepsis RCTs.

## Project layout

```
data/raw/          PDFs go here (gitignored)
data/extracted/    one JSON per paper, output of extraction pipeline
src/ingest.py      PDF → text + section chunks
src/extractor.py   LLM extraction → SepsisStudy JSON
src/validator.py   Pydantic schema + range checks
src/store.py       load/save/query extracted JSONs
src/query.py       NL query → filtered evidence table
src/app.py         Streamlit UI
tests/             pytest suite
config.py          API keys, provider toggle
```

## Provider toggle

Set `USE_OPENROUTER=true` in `.env` to route via OpenRouter.
Set `USE_OPENROUTER=false` to call Anthropic SDK directly.

## Run

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Test

```bash
pytest tests/
```

## Current Architecture

### Files

| File | Role |
|------|------|
| `src/ingest.py` | Section-aware PDF parsing (used by extractor) |
| `src/ingest_enhanced.py` | Two-column layout detection + table extraction (used by RAG) |
| `src/extractor.py` | LLM extraction via OpenRouter/Anthropic → `data/extracted/*.json` |
| `src/validate_enhanced.py` | Confidence scoring: high/medium/low tiering |
| `src/table_builder.py` | Evidence table + AUC-ranked predictor table + keyword filter |
| `src/retrieve.py` | Semantic RAG via `all-MiniLM-L6-v2` embeddings |
| `src/pipeline.py` | Unified entry point: keyword-first, RAG fallback |
| `src/store.py` | Loads extracted JSONs |
| `src/query.py` | Keyword search over store |
| `src/validator.py` | Basic validation |
| `src/app.py` | Streamlit UI (not yet implemented) |
| `run_batch.py` | Batch extraction runner |

### Data

- `data/raw/` — 30 PDFs
- `data/extracted/` — 27 JSONs (3 failed ingestion)
- 217 total predictor records: 173 high confidence, 0 unverified

### Pipeline flow

1. `pipeline.run_pipeline(query, use_case)` called
2. Keyword filter on validated records from `data/extracted/`
3. If ≥3 hits → `build_evidence_table()` or `build_ranked_predictors()`
4. If <3 hits → RAG fallback via `retrieve.py` over raw PDF chunks
5. Returns pandas DataFrame

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

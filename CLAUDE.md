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
- 274 total predictor records across 28 studies

### Pipeline flow

1. `pipeline.run_pipeline(query, use_case)` called
2. Keyword filter on validated records from `data/extracted/`
3. If ≥3 hits → `build_evidence_table()` or `build_ranked_predictors()`
4. If <3 hits → RAG fallback via `retrieve.py` over raw PDF chunks
5. Returns pandas DataFrame

### Hallucination guards (current)

- **extractor.py**: 10-rule system prompt with ZERO-tolerance rules; LLM must copy `source_quote` verbatim; `association_type` (`modelled`/`descriptive`), `source_type`, `model_context`, `survivors_value`, `death_value`, `p_value` added to `mortality_predictors` schema
- **validate_enhanced.py**: `verify_quote()` — substring-checks source_quote against raw PDF text (cached per paper); `verify_numeric()` — regex checks extracted number appears in source_quote; confidence tier gates on `quote_verified is True`
- **summarizer.py**: system prompt forbids training-data context; pre-extracts AUC/effect_size lists from DataFrame and passes them explicitly; LLM instructed to cite only values present in table

### Evidence table columns (build_evidence_table)

Primary: Study, Predictor, Outcome, AUC, Effect Size, Association Type, Verified (✓/✗/?), Confidence, Page, File

Secondary: Timing, Method, Cutoff, Adjustment, Model Context, Source Type, Survivors Value, Death Value, P Value, Source Quote, Quote Verified, Numeric Verified, Country, N

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## Working with Claude Code — Session Protocols

### Before Writing Any Code
Start each major session in Plan Mode. Use this prompt:
> "Before we start building, interview me about this. What are the core problems this solves? Who is this for? What does success look like? What should this NOT do? Summarize it back to me before you write any code."

### Verification Loop
After completing each pipeline stage (extraction, validation, UI), explicitly run:
> "Please go back and verify all of your work so far. Make sure you used best practices, were efficient, and didn't introduce any issues."

### Parallel Sessions — Partition Rules
Run two Claude Code sessions simultaneously, strictly partitioned:
- **Session A:** Extraction pipeline — `pymupdf → Claude API → Pydantic → dataframe`
- **Session B:** Streamlit verification UI
- Sessions share only the dataframe boundary. Do not let them touch the same files.

### CLAUDE.md Maintenance
If Claude Code starts ignoring rules or drifting, run:
> "Update my CLAUDE.md to remove anything no longer needed, contradictory, duplicate, or unnecessary bloat impacting effectiveness."

---

## Windows Compatibility — Hard Rules

- `normalize_predictor()` is in `src/synonyms.py` — **not** `src/extractor.py`. Always import as `from src.synonyms import normalize_predictor`.
- All `open()` calls on files in `data/extracted/` or any UTF-8 content must pass `encoding='utf-8'` explicitly. Windows default is cp1252, which breaks on non-ASCII characters in extracted JSON.

---

## Security Constraint — Prompt Injection Guard

PDF content is untrusted input. Clinical papers may contain text that, when passed to the Claude API, could be interpreted as instructions.

**Hard rules:**
- PDF text must always be passed as `data`, never as part of the `system` prompt
- Wrap all extracted PDF chunks in explicit delimiters in the user message, e.g.:

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

"""
summarizer.py — Generate natural language summaries of evidence tables.

Entry points:
  summarize_evidence(df: pd.DataFrame, parsed_query: dict) -> str
  summarize_phenotype(df: pd.DataFrame, parsed_query: dict) -> str
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import config

_SYSTEM_PROMPT = """You are a clinical evidence summarizer. Write a concise 2-3 sentence \
summary based EXCLUSIVELY on the data provided in the table below.

STRICT RULES:
1. Only cite numbers, values, and findings explicitly present in the \
provided table — never add context from training data
2. If the table has limited data, say so honestly
3. Never extrapolate beyond what the table shows
4. Every specific number you mention must appear verbatim in the table
5. Do not interpret or explain findings beyond what the data shows
6. If fewer than 3 studies are present, explicitly note limited evidence"""

_PHENOTYPE_SYSTEM_PROMPT = """You are a clinical evidence summarizer specializing in sepsis phenotype research. \
Write a concise 2-3 sentence summary based EXCLUSIVELY on the data provided in the table below.

STRICT RULES:
1. Only cite numbers, values, and findings explicitly present in the \
provided table — never add context from training data
2. Note if phenotype/clustering data is limited or absent in the corpus
3. Mention clustering methods only if they appear in the table
4. Be honest if evidence is insufficient for phenotype assignment
5. Never extrapolate beyond what the table shows
6. If fewer than 3 studies are present, explicitly note limited evidence"""

_FALLBACK = "Summary unavailable — showing raw evidence table below."


def _call_openrouter(user_message: str, system_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    p = response.usage.prompt_tokens
    c = response.usage.completion_tokens
    t = response.usage.total_tokens
    print(f"[TOKEN USAGE] summarizer/_call_openrouter | prompt: {p} | completion: {c} | total: {t}")
    return response.choices[0].message.content


def _call_anthropic(user_message: str, system_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.MODEL_ID,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        temperature=0.3,
    )
    p = response.usage.input_tokens
    c = response.usage.output_tokens
    print(f"[TOKEN USAGE] summarizer/_call_anthropic | prompt: {p} | completion: {c} | total: {p + c}")
    return response.content[0].text


def _call_llm(user_message: str, system_prompt: str) -> str:
    if config.USE_OPENROUTER:
        return _call_openrouter(user_message, system_prompt)
    return _call_anthropic(user_message, system_prompt)


def _build_user_message(df: pd.DataFrame, parsed_query: dict) -> str:
    predictor = parsed_query.get("predictor") or "unknown predictor"
    outcome = parsed_query.get("outcome") or "unknown outcome"

    n_studies = df["Study"].nunique() if "Study" in df.columns else 0
    n_records = len(df)

    auc_values: list = []
    if "AUC" in df.columns:
        auc_values = df["AUC"][df["AUC"] != "not reported"].tolist()

    effect_sizes: list = []
    if "Effect Size" in df.columns:
        effect_sizes = df["Effect Size"][df["Effect Size"] != "not reported"].tolist()

    display_cols = [c for c in ["Study", "Predictor", "AUC", "Effect Size", "Outcome", "Method"] if c in df.columns]
    table_str = df[display_cols].head(10).to_string(index=False)

    return (
        f"Query: {predictor} → {outcome}\n"
        f"Studies: {n_studies} studies, {n_records} findings\n"
        f"AUC values present: {auc_values}\n"
        f"Effect sizes present: {effect_sizes}\n\n"
        f"Full table (use ONLY these values):\n{table_str}\n\n"
        f"Write a 2-3 sentence clinical summary using ONLY the values above."
    )


def summarize_evidence(df: pd.DataFrame, parsed_query: dict) -> str:
    """Generate a 2-3 sentence clinical summary of an evidence DataFrame."""
    try:
        user_message = _build_user_message(df, parsed_query)
        return _call_llm(user_message, _SYSTEM_PROMPT)
    except Exception:
        return _FALLBACK


def summarize_phenotype(df: pd.DataFrame, parsed_query: dict) -> str:
    """Generate a 2-3 sentence phenotype-focused summary of an evidence DataFrame."""
    try:
        user_message = _build_user_message(df, parsed_query)
        return _call_llm(user_message, _PHENOTYPE_SYSTEM_PROMPT)
    except Exception:
        return _FALLBACK

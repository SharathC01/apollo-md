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

_SYSTEM_PROMPT = """You are a clinical evidence summarizer. Given a structured evidence table from sepsis research literature, write a concise 2-3 sentence summary.

Rules:
- Mention the number of studies and key findings
- Include specific effect sizes (AUC, OR, HR) where available
- Note any consistency or conflict across studies
- Be precise and clinical in tone — no marketing language
- Never invent values not present in the table
- If data is sparse, say so honestly"""

_PHENOTYPE_SYSTEM_PROMPT = """You are a clinical evidence summarizer specializing in sepsis phenotype research.
Given a structured evidence table, write a concise 2-3 sentence summary.

Rules:
- Note if phenotype/clustering data is limited or absent in the corpus
- Mention any clustering methods found (k-means, latent class analysis, etc.)
- Be honest if the evidence is insufficient for phenotype assignment
- Clinical tone, no marketing language
- Never invent values not present in the table"""

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
    return response.content[0].text


def _call_llm(user_message: str, system_prompt: str) -> str:
    if config.USE_OPENROUTER:
        return _call_openrouter(user_message, system_prompt)
    return _call_anthropic(user_message, system_prompt)


def _build_user_message(df: pd.DataFrame, parsed_query: dict) -> str:
    predictor = parsed_query.get("predictor") or "unknown predictor"
    outcome = parsed_query.get("outcome") or "unknown outcome"

    available_cols = [c for c in ["Study", "Predictor", "AUC", "Effect Size", "Outcome"] if c in df.columns]
    table_str = df[available_cols].head(8).to_string(index=False)

    return (
        f"Query: {predictor} → {outcome}\n"
        f"Studies: {df['Study'].nunique() if 'Study' in df.columns else 'unknown'} studies, {len(df)} findings\n"
        f"Top findings (up to 8 rows):\n{table_str}"
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

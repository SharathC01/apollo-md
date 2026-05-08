"""
query_parser.py — Parse natural language clinical queries into structured search intent.

Entry point:
  parse_query(query: str) -> dict

Returns:
  {
    "predictor": str | None,
    "outcome": str | None,
    "population": str | None,
    "time_horizon": str | None,
    "keywords": list[str],
  }
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

_SYSTEM_PROMPT = """You are a clinical query parser. Extract structured search intent from a natural language clinical question about sepsis research.

Return ONLY valid JSON with these fields:
{
  "predictor": string or null,
  "outcome": string or null,
  "population": string or null,
  "time_horizon": string or null,
  "keywords": [string]
}

Rules:
- Return null for fields not mentioned in the query
- keywords must be single words or short phrases likely to appear in paper text
- Include 3-5 keywords
- Never infer or guess — only extract what is explicitly stated or strongly implied"""


def _call_openrouter(user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )
    response = client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    p = response.usage.prompt_tokens
    c = response.usage.completion_tokens
    t = response.usage.total_tokens
    print(f"[TOKEN USAGE] query_parser/_call_openrouter | prompt: {p} | completion: {c} | total: {t}")
    return response.choices[0].message.content


def _call_anthropic(user_message: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=config.MODEL_ID,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        temperature=0,
    )
    p = response.usage.input_tokens
    c = response.usage.output_tokens
    print(f"[TOKEN USAGE] query_parser/_call_anthropic | prompt: {p} | completion: {c} | total: {p + c}")
    return response.content[0].text


def _call_llm(user_message: str) -> str:
    if config.USE_OPENROUTER:
        return _call_openrouter(user_message)
    return _call_anthropic(user_message)


def _fallback(query: str) -> dict:
    return {
        "predictor": None,
        "outcome": None,
        "population": None,
        "time_horizon": None,
        "keywords": query.split(),
    }


def parse_query(query: str) -> dict:
    """Parse a natural language clinical query into structured search intent."""
    try:
        raw = _call_llm(query)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
    except Exception:
        return _fallback(query)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "SOFA score and 28-day mortality"
    import pprint
    pprint.pprint(parse_query(query))

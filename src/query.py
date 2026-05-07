"""
query.py — Natural language query → evidence table.

Responsibilities:
  - Accept a free-text clinical question (e.g., "mortality in early antibiotics trials")
  - Use LLM to map question to filter criteria over SepsisStudy fields
  - Apply filters to loaded studies (via store.py)
  - Return ranked evidence table as pandas DataFrame
  - Support follow-up queries that narrow previous results

Entry points:
  nl_query(question: str, studies: list[dict]) -> pd.DataFrame
"""

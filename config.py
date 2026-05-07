"""
config.py — Central config loader.

Reads from .env:
  OPENROUTER_API_KEY  — key for OpenRouter API
  ANTHROPIC_API_KEY   — key for Anthropic API
  USE_OPENROUTER      — "true"/"false" toggle (default: true)

Exposes:
  USE_OPENROUTER: bool
  OPENROUTER_API_KEY: str
  ANTHROPIC_API_KEY: str
  MODEL_ID: str          — resolved model string for active provider
"""

from dotenv import load_dotenv
import os

load_dotenv()

USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "true").lower() == "true"
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Model IDs per provider
_OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"
_ANTHROPIC_MODEL = "claude-sonnet-4-6"

MODEL_ID: str = _OPENROUTER_MODEL if USE_OPENROUTER else _ANTHROPIC_MODEL

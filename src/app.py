"""
app.py — Streamlit UI.

Responsibilities:
  - Sidebar: PDF upload → trigger ingest + extract pipeline
  - Main panel: NL query box → evidence table display
  - Study browser: filterable table of all extracted studies
  - Detail view: click row → show full SepsisStudy fields
  - Export: download filtered results as CSV

Run:
  streamlit run src/app.py
"""

import streamlit as st

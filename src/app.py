"""
app.py — Apollo MD Streamlit UI.
Sepsis evidence extraction and synthesis interface.
"""

import math
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import run_pipeline, get_source_quote

# Must be first Streamlit call
st.set_page_config(
    page_title="Apollo MD",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ────────────────────────────────────────────────────────────────
BADGE_TEXT = "28 papers · 360 records · 247 verified"

EXAMPLE_QUERIES = {
    "mortality": "What predicts 28-day mortality in septic shock?",
    "phenotype": "What sepsis phenotypes have been identified in ICU patients?",
    "biomarker": "Which biomarker best predicts sepsis mortality?",
}

DISPLAY_COLUMNS = [
    "Study", "Predictor", "Outcome", "AUC", "Effect Size",
    "Association Type", "Verified", "Confidence", "Page", "File",
]

_CONF_EMOJI = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}
_GRADE_MAP = {
    "high": "⬛⬛⬛⬛ High",
    "medium": "⬛⬛⬛⬜ Moderate",
    "low": "⬛⬛⬜⬜ Low",
}
_CONF_SORT = {"🟢 High": 0, "🟡 Medium": 1, "🔴 Low": 2}


# ── Caching ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def cached_pipeline(query: str, use_case: str):
    df, summary = run_pipeline(query, use_case)
    return df, summary


# ── Pure helpers ─────────────────────────────────────────────────────────────
def _auc_float(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, float):
        return None if math.isnan(val) else val
    if isinstance(val, int):
        return float(val)
    s = str(val)
    if s in ("not reported", "nan", ""):
        return None
    m = re.search(r"(\d+\.\d+)", s)
    return float(m.group(1)) if m else None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unified column schema; fill missing with 'not reported'."""
    df = df.copy()
    # biomarker structured path: AUC is numeric, no AUC (full) issue
    if "AUC (full)" in df.columns and "AUC" not in df.columns:
        df["AUC"] = df["AUC (full)"]
    needed = [
        "Study", "Country", "N", "Predictor", "Outcome", "Timing", "Method",
        "Effect Size", "AUC", "Cutoff", "Adjustment", "Verified", "Confidence",
        "Source Quote", "Page", "File", "result_type",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = "not reported"
    return df


def _enrich(df: pd.DataFrame, use_case: str) -> pd.DataFrame:
    df = _normalize(df)

    def _design(m: str) -> str:
        s = str(m).lower()
        if "randomi" in s:
            return "RCT 🟩"
        if "meta-analysis" in s or "systematic" in s:
            return "Meta 🟦"
        if any(k in s for k in ("roc", "auc", "logistic", "cox", "observ")):
            return "Obs 🟨"
        return "— ⬜"

    df["Design"] = df["Method"].apply(_design)
    df["Evidence Grade"] = df["Confidence"].apply(
        lambda c: _GRADE_MAP.get(str(c).lower(), "⬜⬜⬜⬜ Unknown")
    )

    def _pico(row) -> str:
        base = f"P: ICU sepsis · I: {row['Predictor']} · O: {row['Outcome']}"
        t = str(row.get("Timing", ""))
        if t and t not in ("not reported", "nan", ""):
            base += f" · T: {t}"
        return base

    df["PICO"] = df.apply(_pico, axis=1)

    # Transform Confidence to emoji after Evidence Grade uses raw value
    df["Confidence"] = df["Confidence"].apply(
        lambda c: _CONF_EMOJI.get(str(c).lower(), str(c))
    )

    # Sort
    if use_case == "biomarker":
        df["_s"] = df["AUC"].apply(_auc_float).fillna(-1.0)
        df = df.sort_values("_s", ascending=False).drop(columns=["_s"])
    else:
        df["_c"] = df["Confidence"].map(lambda c: _CONF_SORT.get(c, 99))
        df["_a"] = df["AUC"].apply(_auc_float).fillna(-1.0)
        df = df.sort_values(["_c", "_a"], ascending=[True, False]).drop(columns=["_c", "_a"])

    return df.reset_index(drop=True)


def _het_check(df: pd.DataFrame) -> list[str]:
    flagged = []
    for pred, grp in df.groupby("Predictor"):
        if len(grp) < 3:
            continue
        aucs = grp["AUC"].apply(_auc_float).dropna()
        if len(aucs) >= 2 and (aucs.max() - aucs.min()) > 0.15:
            flagged.append(str(pred))
    return flagged


def _summary_line(df: pd.DataFrame) -> str:
    n = len(df)
    n_high = int((df["Confidence"] == "🟢 High").sum())
    n_c = df["Country"].nunique() if "Country" in df.columns else 0
    aucs = df["AUC"].apply(_auc_float).dropna()
    s = f"Returning {n} records · {n_high} high confidence · {n_c} countries"
    if len(aucs) >= 2:
        s += f" · AUC range: {aucs.min():.2f}–{aucs.max():.2f}"
    return s


def _citations(df: pd.DataFrame) -> str:
    lines = []
    for study in df["Study"].dropna().unique():
        parts = str(study).split("_")
        author = parts[0]
        year = parts[1] if len(parts) > 1 else "n.d."
        lines.append(f"{author} et al. ({year}). Sepsis Evidence Record. Apollo MD.")
    return "\n".join(lines)


# ── Session state init ───────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "query": "",
    "use_case": "mortality",
    "results_df": None,
    "summary": "",
    "selected_row": None,
    "full_quote": None,
    "full_quote_key": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Transfer pending query before widget instantiation
if "query_pending" in st.session_state:
    st.session_state["query"] = st.session_state.pop("query_pending")

# ── Header ───────────────────────────────────────────────────────────────────
_h1, _h2 = st.columns([4, 1])
with _h1:
    st.markdown("## **Apollo MD** — *Sepsis Evidence Engine*")
with _h2:
    st.markdown(f"<br>`{BADGE_TEXT}`", unsafe_allow_html=True)
st.divider()

# ── Query input ──────────────────────────────────────────────────────────────
st.text_input(
    "Query",
    key="query",
    placeholder="e.g. What predicts 28-day mortality in septic shock?",
    label_visibility="collapsed",
)
st.caption(
    "Tip: queries naming a specific predictor (e.g. SOFA, lactate) return results "
    "in ~3s. Broad queries use semantic search (~15s)."
)


# ── Use case tabs ─────────────────────────────────────────────────────────────
def _tab_body(uc: str) -> bool:
    """Render chip + Run button. Chip click sets query + pending run, then reruns."""
    eq = EXAMPLE_QUERIES[uc]
    label = eq if len(eq) <= 55 else eq[:52] + "…"
    c1, c2 = st.columns([5, 1])
    with c1:
        if st.button(f'"{label}"', key=f"chip_{uc}", help=eq):
            st.session_state["query_pending"] = eq
            st.session_state["_pending_uc"] = uc
            st.rerun()
    with c2:
        return st.button("Run Query", type="primary", key=f"run_{uc}")


_tab_mort, _tab_pheno, _tab_bio = st.tabs([
    "UC1 · Mortality Predictors",
    "UC2 · Phenotype Extraction",
    "UC3 · Biomarker Ranking",
])

_run_uc: str | None = None

with _tab_mort:
    if _tab_body("mortality"):
        _run_uc = "mortality"
with _tab_pheno:
    if _tab_body("phenotype"):
        _run_uc = "phenotype"
with _tab_bio:
    if _tab_body("biomarker"):
        _run_uc = "biomarker"

# Chip-triggered pending run
_pending = st.session_state.get("_pending_uc")
if _pending is not None:
    del st.session_state["_pending_uc"]
    _run_uc = _pending


# ── Pipeline execution ────────────────────────────────────────────────────────
if _run_uc is not None:
    _q = st.session_state["query"]
    if _q:
        st.session_state["use_case"] = _run_uc
        st.session_state["selected_row"] = None
        st.session_state["full_quote"] = None
        st.session_state["full_quote_key"] = None
        with st.spinner("Running query... (semantic search may take ~15s)"):
            try:
                _raw, _sum = cached_pipeline(_q, _run_uc)
                st.session_state["results_df"] = _raw
                st.session_state["summary"] = _sum
            except Exception as _e:
                st.error(f"Pipeline error: {str(_e)}")
                st.session_state["results_df"] = None
                st.session_state["summary"] = ""


# ── Results ───────────────────────────────────────────────────────────────────
_raw_df: pd.DataFrame | None = st.session_state["results_df"]
_summary: str = st.session_state["summary"] or ""
_use_case: str = st.session_state["use_case"]

if _raw_df is not None:
    if len(_raw_df) == 0:
        st.info(
            "No records matched. Try a more specific predictor name, or switch use case."
        )
        _eq = EXAMPLE_QUERIES[_use_case]
        if st.button(f'Try: "{_eq}"', key="empty_chip"):
            st.session_state["query"] = _eq
            st.session_state["_pending_uc"] = _use_case
            st.rerun()
    else:
        _df = _enrich(_raw_df, _use_case)
        _rt = (
            _raw_df["result_type"].iloc[0]
            if "result_type" in _raw_df.columns
            else "structured"
        )

        if _rt == "structured":
            st.success(f"Structured match — {len(_df)} records found for this predictor.")
            st.caption("✓ Structured match — returned in ~3s")
        else:
            st.caption("⏱ Semantic search — embedding step took ~15s")

        _col_l, _col_r = st.columns([65, 35])

        # ── Left column ───────────────────────────────────────────────────────
        with _col_l:
            if _summary:
                with st.expander("📋 Evidence Synthesis", expanded=True):
                    st.text(_summary)

            st.caption(_summary_line(_df))

            _het = _het_check(_df)
            if _het:
                st.warning(
                    f"⚠ Heterogeneity detected in {len(_het)} predictor(s): "
                    f"{', '.join(_het)}\n\n"
                    "Effect sizes vary substantially across studies. "
                    "Interpret aggregated results with caution."
                )

            if _rt == "semantic":
                st.info(
                    "⚠ Semantic search results — no predictor matched in query. "
                    "Showing top 5 corpus passages. "
                    "Structured extraction unavailable for this query."
                )
                if _use_case == "phenotype":
                    st.caption(
                        "Phenotyping papers are underrepresented in the current corpus. "
                        "Results reflect best available semantic evidence."
                    )

            if _use_case == "biomarker":
                _chart = _df[["Predictor", "AUC"]].copy()
                _chart["_v"] = _chart["AUC"].apply(_auc_float)
                _chart = _chart.dropna(subset=["_v"]).sort_values("_v")
                if not _chart.empty:
                    _fig = px.bar(
                        _chart,
                        x="_v",
                        y="Predictor",
                        orientation="h",
                        title="Biomarkers Ranked by AUC",
                        labels={"_v": "AUC", "Predictor": ""},
                    )
                    st.plotly_chart(_fig, use_container_width=True)

            _show_cols = [c for c in DISPLAY_COLUMNS if c in _df.columns]
            if _use_case == "biomarker" and "AUC" in _show_cols:
                _show_cols = ["AUC"] + [c for c in _show_cols if c != "AUC"]

            _ev = st.dataframe(
                _df[_show_cols],
                selection_mode="single-row",
                use_container_width=True,
                height=450,
                on_select="rerun",
                key="results_table",
            )

            _sel_rows = getattr(getattr(_ev, "selection", None), "rows", [])
            if _sel_rows:
                _new_sel = _sel_rows[0]
                if _new_sel != st.session_state["selected_row"]:
                    st.session_state["selected_row"] = _new_sel
                    st.session_state["full_quote"] = None
                    st.session_state["full_quote_key"] = None

            _ts = datetime.now().strftime("%Y%m%d_%H%M")
            _e1, _e2, _e3 = st.columns(3)
            with _e1:
                st.download_button(
                    "⬇ Download Evidence Table (CSV)",
                    data=_df.to_csv(index=False),
                    file_name=f"apollo_md_evidence_{_use_case}_{_ts}.csv",
                    mime="text/csv",
                    key="dl_csv",
                )
            with _e2:
                st.download_button(
                    "📋 Copy Citation List",
                    data=_citations(_df),
                    file_name=f"apollo_md_citations_{_ts}.txt",
                    mime="text/plain",
                    key="dl_cite",
                )
            with _e3:
                _gcols = [
                    c for c in
                    ["Study", "Predictor", "AUC", "Evidence Grade", "Verified", "N"]
                    if c in _df.columns
                ]
                st.download_button(
                    "⬇ Export GRADE Table (TSV)",
                    data=_df[_gcols].to_csv(index=False, sep="\t"),
                    file_name=f"apollo_md_grade_{_use_case}_{_ts}.tsv",
                    mime="text/tab-separated-values",
                    key="dl_grade",
                )
                st.caption("Compatible with GRADE GDT tool format")

        # ── Right column ──────────────────────────────────────────────────────
        with _col_r:
            _sel_idx = st.session_state["selected_row"]
            if _sel_idx is None or _sel_idx >= len(_df):
                st.markdown("*← Select a row to inspect its source evidence*")
            else:
                _row = _df.iloc[_sel_idx]
                _study   = str(_row.get("Study",       ""))
                _file    = str(_row.get("File",        "not reported"))
                _page    = _row.get("Page",            "not reported")
                _verified = str(_row.get("Verified",   ""))
                _conf    = str(_row.get("Confidence",  ""))
                _pred    = str(_row.get("Predictor",   ""))
                _fx      = str(_row.get("Effect Size", "not reported"))
                _auc_v   = str(_row.get("AUC",         "not reported"))
                _meth    = str(_row.get("Method",      "not reported"))
                _adj     = str(_row.get("Adjustment",  "not reported"))
                _sq      = str(_row.get("Source Quote",""))
                _pico_v  = str(_row.get("PICO",        ""))
                _raw_conf = (
                    _conf.replace("🟢 ", "")
                         .replace("🟡 ", "")
                         .replace("🔴 ", "")
                         .lower().strip()
                )

                st.markdown("── Source Evidence ──────────────────────")
                st.markdown(f"**Study:** {_study}")
                st.markdown(f"**File:** {_file} · Page {_page}")
                st.markdown(f"**Verified:** {_verified} · {_conf} confidence")

                _qkey = f"{_study}_{_pred}"
                _fq_valid = st.session_state.get("full_quote_key") == _qkey
                _disp_q = (
                    st.session_state["full_quote"]
                    if _fq_valid and st.session_state["full_quote"]
                    else _sq
                )
                st.info(_disp_q or "Source quote not available.")

                if not (_fq_valid and st.session_state.get("full_quote")):
                    if st.button("Load full source quote", key=f"load_fq_{_sel_idx}"):
                        _full = get_source_quote(_study, _pred)
                        st.session_state["full_quote"] = _full
                        st.session_state["full_quote_key"] = _qkey
                        st.rerun()

                st.divider()
                st.markdown(f"**Predictor:** {_pred}")
                st.markdown(f"**Effect Size:** {_fx}")
                st.markdown(f"**AUC:** {_auc_v}")
                st.markdown(f"**Method:** {_meth}")
                st.markdown(f"**Adjustment:** {_adj}")
                st.caption(_pico_v)

                st.markdown("── Honest Uncertainty ────────────────────")
                if _fx == "not reported":
                    st.caption("Effect size not reported. AUC used as primary metric.")
                if _raw_conf == "low":
                    st.caption("Low confidence extraction. Manual verification recommended.")
                if _adj == "not reported":
                    st.caption("Confounders not reported. Interpret effect size with caution.")

                st.caption(
                    "Note: Only modelled associations (OR/HR/AUC) are captured. "
                    "Descriptive comparisons (median differences) are not extracted."
                )

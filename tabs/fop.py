# tabs/fop.py
from __future__ import annotations

import io

import pandas as pd
import requests
import streamlit as st

from tabs.base_intel import IntelConfig, render_intel
from utils.services import get_http_session


DEFAULT_QUERIES = [
    "fractional ownership real estate platforms India",
    "fractional real estate investment platforms India",
    "shared property ownership platforms India",
    "real estate crowdfunding India",
    "real estate investment platforms fractional ownership India",
    "property fractionalization platforms India",
    "fractional ownership property India",
    "real estate tokenization India",
    "SEBI approved fractional ownership platforms India",
]


def fop_relevance_filter(title: str, snippet: str, query: str) -> bool:
    """FOP-oriented relevance filter."""

    text = (str(title) + " " + str(snippet)).lower()

    fop_terms = [
        "fractional",
        "ownership",
        "real estate",
        "property",
        "reit",
        "sm reit",
        "crowdfunding",
        "tokenization",
        "commercial",
        "residential",
        "co-ownership",
    ]
    platform_terms = ["platform", "app", "website", "portal", "online", "provider", "invest", "review", "login"]

    has_context = any(t in text for t in fop_terms)
    has_platform = any(t in text for t in platform_terms)

    brands = ["strata", "hbits", "property share", "myre", "alt drx", "wisex", "aerem", "alyf"]
    query_lower = (query or "").lower()
    if any(b in query_lower for b in brands):
        return any(b in text for b in brands)

    return has_context and has_platform


@st.cache_data(ttl=3600)
def fetch_fop_registry() -> pd.DataFrame:
    """Fetches a REIT-related SEBI list (best-effort, depends on SEBI export format)."""

    session = get_http_session()
    registry_list = []

    try:
        sebi_url = "https://www.sebi.gov.in/sebiweb/other/IntmExportAction.do?intmId=48"
        r = session.get(sebi_url, timeout=20)
        if r.status_code != 200:
            return pd.DataFrame(registry_list)

        # The export is sometimes Excel, sometimes HTML.
        df = None
        try:
            df = pd.read_excel(io.BytesIO(r.content))
        except Exception:
            try:
                dfs = pd.read_html(r.content)
                if dfs:
                    df = dfs[0]
            except Exception:
                df = None

        if df is None or df.empty:
            return pd.DataFrame(registry_list)

        df.columns = [str(c).lower().strip() for c in df.columns]

        name_col = next((c for c in df.columns if "name" in c), None)
        reg_col = next((c for c in df.columns if "registration" in c), None)

        if name_col and reg_col:
            for _, row in df.iterrows():
                registry_list.append(
                    {
                        "Entity Name": str(row.get(name_col, "")).strip(),
                        "SEBI ID": str(row.get(reg_col, "")).strip(),
                        "Website": "-",
                        "Source": "SEBI Export (intmId=48)",
                    }
                )

    except Exception:
        pass

    out = pd.DataFrame(registry_list)
    if not out.empty:
        out = out.dropna(subset=["Entity Name"]).drop_duplicates(subset=["Entity Name", "SEBI ID"]).reset_index(drop=True)
    return out


def render() -> None:
    cfg = IntelConfig(
        key="fop",
        header="FOP Market Intelligence",
        description="Search fractional real-estate articles, extract platform names, and verify their SEBI status (where applicable).",
        default_queries=DEFAULT_QUERIES,
        relevance_filter=fop_relevance_filter,
        registry_loader=fetch_fop_registry,
        entity_type="Fractional ownership / real estate investment platforms",
        verify_suffix='SEBI registration (REIT OR "SM REIT")',
        id_regexes=[
            r"IN\/(REIT|SM\s?REIT)\/\d{2}-\d{2}\/\d{3,6}",
            r"IN\/[A-Z\s]{3,10}\/\d{2}-\d{2}\/\d{3,6}",
        ],
    )

    render_intel(cfg)
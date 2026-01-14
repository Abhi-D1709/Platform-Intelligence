# tabs/obpp.py
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests
import streamlit as st

from tabs.base_intel import IntelConfig, render_intel
from utils.services import HEADERS, get_http_session


DEFAULT_QUERIES = [
    "bonds investment platform India",
    "invest in bonds India",
    "invest in bonds online India",
    "fixed income investing India",
    "buying bonds India",
    "investment in debt securities India",
]


def obpp_relevance_filter(title: str, snippet: str, query: str) -> bool:
    """Bond/OBPP-oriented relevance filter (excludes fractional real estate)."""

    text = (str(title) + " " + str(snippet)).lower()

    # Exclude FOP-ish contexts
    if "fractional" in text or "real estate" in text or "reit" in text:
        return False

    finance_terms = ["bond", "debenture", "yield", "debt", "fixed income", "g-sec", "treasury", "ncd"]
    platform_terms = ["platform", "app", "website", "portal", "online", "provider", "review", "best", "invest"]

    has_finance = any(t in text for t in finance_terms)
    has_platform = any(t in text for t in platform_terms)

    # Optional brand allowlist for some queries
    query_lower = (query or "").lower()
    brands = ["wint", "goldenpi", "indiabonds", "zerodha", "grip"]
    if any(b in query_lower for b in brands):
        return any(b in text for b in brands)

    return has_finance and has_platform


@st.cache_data(ttl=3600)
def fetch_obpp_registry() -> pd.DataFrame:
    """Downloads and parses official OBPP lists from NSE and BSE."""

    registry_list = []
    session = get_http_session()

    # NSE (XLSX)
    try:
        nse_url = "https://nsearchives.nseindia.com/web/sites/default/files/inline-files/List_of_Registered_OBPPs_NSE.xlsx"
        r = session.get(nse_url, timeout=15)
        if r.status_code == 200:
            df_nse = pd.read_excel(io.BytesIO(r.content))
            df_nse.columns = [str(c).lower().strip() for c in df_nse.columns]

            name_col = next((c for c in df_nse.columns if "name" in c and "member" in c), None)
            reg_col = next((c for c in df_nse.columns if "sebi" in c or "registration" in c), None)
            web_col = next((c for c in df_nse.columns if "website" in c), None)

            if name_col and reg_col:
                for _, row in df_nse.iterrows():
                    registry_list.append(
                        {
                            "Entity Name": str(row.get(name_col, "")).strip(),
                            "SEBI ID": str(row.get(reg_col, "")).strip(),
                            "Website": str(row.get(web_col, "-")).strip() if web_col else "-",
                            "Source": "NSE Official List",
                        }
                    )
    except Exception:
        pass

    # BSE (ZIP -> XLSX/CSV)
    try:
        bse_url = "https://www.bseindia.com/downloads1/OBP_MEMBER_LIST.zip"
        r = session.get(bse_url, timeout=15)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for filename in z.namelist():
                    if not filename.lower().endswith((".xlsx", ".xls", ".csv")):
                        continue

                    with z.open(filename) as f:
                        if filename.lower().endswith(".csv"):
                            df_bse = pd.read_csv(f)
                        else:
                            df_bse = pd.read_excel(f)

                    df_bse.columns = [str(c).lower().strip() for c in df_bse.columns]
                    name_col = next((c for c in df_bse.columns if "name" in c), None)
                    reg_col = next((c for c in df_bse.columns if "sebi" in c or "registration" in c), None)

                    if name_col and reg_col:
                        for _, row in df_bse.iterrows():
                            registry_list.append(
                                {
                                    "Entity Name": str(row.get(name_col, "")).strip(),
                                    "SEBI ID": str(row.get(reg_col, "")).strip(),
                                    "Website": "-",
                                    "Source": "BSE Official List",
                                }
                            )
                    break
    except Exception:
        pass

    df = pd.DataFrame(registry_list)
    if not df.empty:
        df = df.dropna(subset=["Entity Name"]).drop_duplicates(subset=["Entity Name", "SEBI ID"]).reset_index(drop=True)
    return df


def render() -> None:
    cfg = IntelConfig(
        key="obpp",
        header="Bond Market Intelligence",
        description="Search bond-related articles, extract platform names, and verify their SEBI registration.",
        default_queries=DEFAULT_QUERIES,
        relevance_filter=obpp_relevance_filter,
        registry_loader=fetch_obpp_registry,
        entity_type="Online Bond Platforms (OBPPs) / Bond investment platforms",
        verify_suffix='SEBI registration INZ OR ("online bond platform")',
        id_regexes=[r"INZ\d{6,}"],
    )

    render_intel(cfg)
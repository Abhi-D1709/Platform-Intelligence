# tabs/base_intel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.common import render_page_header
from utils.services import (
    extract_entities_with_ai,
    fetch_content_batch,
    generic_sebi_verifier,
    run_duckduckgo_search,
)


@dataclass(frozen=True)
class IntelConfig:
    key: str  # session prefix: "obpp" / "fop" etc.
    header: str
    description: str
    default_queries: List[str]
    relevance_filter: Callable[[str, str, str], bool]
    registry_loader: Callable[[], pd.DataFrame]
    entity_type: str
    verify_suffix: str
    id_regexes: List[str]


def _k(cfg: IntelConfig, suffix: str) -> str:
    return f"{cfg.key}:{suffix}"


def _ensure_defaults(cfg: IntelConfig) -> None:
    st.session_state.setdefault(_k(cfg, "lite_mode"), True)
    st.session_state.setdefault(_k(cfg, "num_results"), 10)
    st.session_state.setdefault(_k(cfg, "queries"), "\n".join(cfg.default_queries))
    st.session_state.setdefault(_k(cfg, "selected_indices"), [])


def _effective_api_key(user_value: str) -> str:
    user_value = (user_value or "").strip()
    if user_value:
        return user_value
    return (st.secrets.get("GEMINI_API_KEY", "") or "").strip()


def _count_mentions_by_source(entities: Sequence[str], content_list: Sequence[dict]) -> pd.DataFrame:
    rows = []
    for e in entities:
        e_low = e.lower()
        mentions = 0
        for item in content_list:
            hay = (item.get("title", "") + " " + item.get("content", "")).lower()
            if e_low in hay:
                mentions += 1
        rows.append({"Platform Name": e, "Mentions": mentions})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Mentions", "Platform Name"], ascending=[False, True]).reset_index(drop=True)
    return df


def render_intel(cfg: IntelConfig) -> None:
    _ensure_defaults(cfg)

    render_page_header(cfg.header, cfg.description)

    debug_log: List[str] = []

    # --- Sidebar (use a form to reduce accidental reruns) ---
    with st.sidebar:
        st.header("Configuration")

        with st.form(_k(cfg, "config_form")):
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                value="",
                help="If left blank, the app will use GEMINI_API_KEY from st.secrets (if configured).",
            )

            lite_mode = st.checkbox(
                "Lite Mode (Faster)",
                value=st.session_state[_k(cfg, "lite_mode")],
                help="Uses snippets instead of full page scraping.",
            )

            num_results = st.slider(
                "Results per query",
                min_value=5,
                max_value=50,
                value=int(st.session_state[_k(cfg, "num_results")]),
            )

            queries_input = st.text_area(
                "Queries",
                value=st.session_state[_k(cfg, "queries")],
                height=160,
            )

            apply = st.form_submit_button("Apply")

        if apply:
            st.session_state[_k(cfg, "lite_mode")] = bool(lite_mode)
            st.session_state[_k(cfg, "num_results")] = int(num_results)
            st.session_state[_k(cfg, "queries")] = queries_input

        api_key = _effective_api_key(api_key_input)
        if api_key and not api_key_input:
            st.caption("Using credentials from st.secrets.")

    queries = [q.strip() for q in st.session_state[_k(cfg, "queries")].split("\n") if q.strip()]

    tab1, tab2 = st.tabs(["🔍 Search & Filter", "🧠 AI Extraction & Verification"])

    # --- TAB 1: Search ---
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Step 1: Run a search to find relevant articles.")
        with col2:
            if st.button("Run Search", type="primary", use_container_width=True, key=_k(cfg, "run_search")):
                with st.spinner("Searching..."):
                    df = run_duckduckgo_search(
                        queries,
                        filter_func=cfg.relevance_filter,
                        num_results=int(st.session_state[_k(cfg, "num_results")]),
                        region="in-en",
                        debug_log=debug_log,
                    )

                    st.session_state[_k(cfg, "results")] = df
                    st.session_state[_k(cfg, "debug_log")] = "\n".join(debug_log)

                    if df.empty:
                        st.error("No matches found.")
                    else:
                        st.success(f"Found {len(df)} relevant results.")

        df_results: pd.DataFrame = st.session_state.get(_k(cfg, "results"), pd.DataFrame())

        # --- Reset selection if search results changed (prevents stale checkbox keys) ---
        results_token = tuple(df_results["Link"].astype(str).tolist()) if "Link" in df_results.columns else tuple(df_results.index.tolist())
        token_key = _k(cfg, "results_token")
        sel_key = _k(cfg, "selected_indices")

        if st.session_state.get(token_key) != results_token:
            st.session_state[token_key] = results_token
            st.session_state[sel_key] = []

            # Clear old checkbox keys for this module
            prefix = f"{cfg.key}:cb:"
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith(prefix):
                    del st.session_state[k]

        if not df_results.empty:
            st.subheader("Search Results")
            all_domains = sorted(df_results["Domain"].dropna().unique().tolist())
            selected_domains = st.multiselect(
                "Filter by Domain",
                all_domains,
                default=all_domains,
                key=_k(cfg, "domains"),
            )

            filtered_df = df_results[df_results["Domain"].isin(selected_domains)].copy()
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={"Link": st.column_config.LinkColumn("Link")},
            )

            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                file_name=f"{cfg.key}_search_results.csv",
                mime="text/csv",
            )

    # --- TAB 2: AI + Verification ---
    with tab2:
        st.markdown("### Deep Extraction & Verification")
        st.info("Step 2: Extract entities from selected sources and verify against official lists.")

        # Registry cache in session
        registry_key = _k(cfg, "registry")
        if registry_key not in st.session_state:
            with st.spinner("Fetching official registry..."):
                st.session_state[registry_key] = cfg.registry_loader()

        df_results: pd.DataFrame = st.session_state.get(_k(cfg, "results"), pd.DataFrame())
        if df_results.empty:
            st.warning("Please run a search in Tab 1 first.")
            return

        # --- Selection UI (checkbox list + select all) ---
        sel_key = _k(cfg, "selected_indices")
        selected_indices = set(st.session_state.get(sel_key, []))

        st.markdown("**Select articles to analyze:**")

        # Optional filter for long lists
        filter_text = st.text_input(
            "Filter articles (title/domain)",
            value="",
            key=_k(cfg, "article_filter"),
            placeholder="Type to filter…",
        )

        df_view = df_results.copy()
        if filter_text.strip():
            ft = filter_text.strip().lower()
            df_view = df_view[
                df_view["Title"].astype(str).str.lower().str.contains(ft, na=False)
                | df_view["Domain"].astype(str).str.lower().str.contains(ft, na=False)
            ]

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button(f"Select all ({len(df_view)} shown)", key=_k(cfg, "select_all")):
                for idx in df_view.index.tolist():
                    st.session_state[f"{cfg.key}:cb:{idx}"] = True
                st.session_state[sel_key] = df_view.index.tolist()
                st.rerun()

        with col_b:
            if st.button("Clear all", key=_k(cfg, "clear_all")):
                # Clear all checkboxes for current full results (not just filtered)
                for idx in df_results.index.tolist():
                    st.session_state[f"{cfg.key}:cb:{idx}"] = False
                st.session_state[sel_key] = []
                st.rerun()

        # Render checkbox list
        new_selected = set()
        for idx, row in df_view.iterrows():
            label = f"{row.get('Title','')} ({row.get('Domain','')})"
            cb_key = f"{cfg.key}:cb:{idx}"

            checked = st.checkbox(
                label,
                key=cb_key,
                value=(idx in selected_indices),
            )
            if checked:
                new_selected.add(idx)

        st.session_state[sel_key] = sorted(new_selected)
        selected_indices = st.session_state[sel_key]

        if st.button("Extract Platforms", type="primary", key=_k(cfg, "extract")):
            if not api_key:
                st.error("Gemini API Key is required (enter it in the sidebar or configure st.secrets).")
            elif not selected_indices:
                st.warning("Select at least one article.")
            else:
                selected_rows = df_results.loc[selected_indices].to_dict("records")

                with st.status("Running AI extraction...", expanded=True) as status:
                    st.write("Fetching content...")
                    content_list = fetch_content_batch(
                        selected_rows,
                        lite_mode=bool(st.session_state[_k(cfg, "lite_mode")]),
                    )

                    st.write("Analyzing with Gemini...")
                    entities, raw = extract_entities_with_ai(
                        api_key=api_key,
                        content_list=content_list,
                        entity_type=cfg.entity_type,
                    )

                    st.session_state[_k(cfg, "content_list")] = content_list
                    st.session_state[_k(cfg, "entities")] = entities
                    st.session_state[_k(cfg, "ai_raw")] = raw

                    status.update(label="Extraction complete", state="complete", expanded=False)

                if not entities:
                    st.warning("No entities extracted.")
                else:
                    st.success(f"Extracted {len(entities)} unique entities.")

        # Visualize + Verify
        entities: List[str] = st.session_state.get(_k(cfg, "entities"), [])
        content_list: List[dict] = st.session_state.get(_k(cfg, "content_list"), [])

        if entities:
            st.divider()
            st.subheader("Mentions")

            counts_df = _count_mentions_by_source(entities, content_list)

            fig = px.bar(
                counts_df,
                x="Mentions",
                y="Platform Name",
                orientation="h",
                title="Mentions across selected sources",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("Regulatory Check")

            if st.button("Verify SEBI Registration", key=_k(cfg, "verify")):
                registry_df: pd.DataFrame = st.session_state.get(registry_key, pd.DataFrame())

                results = []
                prog = st.progress(0)

                for i, name in enumerate(counts_df["Platform Name"].tolist()):
                    status, reg_id, link = generic_sebi_verifier(
                        platform_name=name,
                        registry_df=registry_df,
                        search_pattern_suffix=cfg.verify_suffix,
                        id_regexes=cfg.id_regexes,
                    )
                    results.append({"Platform": name, "Status": status, "ID": reg_id, "Link": link})
                    prog.progress((i + 1) / max(len(counts_df), 1))

                st.session_state[_k(cfg, "verification")] = pd.DataFrame(results)
                prog.empty()

            ver_df: pd.DataFrame = st.session_state.get(_k(cfg, "verification"), pd.DataFrame())
            if not ver_df.empty:
                st.dataframe(
                    ver_df,
                    use_container_width=True,
                    column_config={"Link": st.column_config.LinkColumn("Proof Link")},
                )

        # Debug downloads
        st.divider()
        raw = st.session_state.get(_k(cfg, "ai_raw"), "")
        logs = st.session_state.get(_k(cfg, "debug_log"), "")

        debug_bundle = []
        if logs:
            debug_bundle.append("--- SEARCH LOGS ---\n" + logs)
        if raw:
            debug_bundle.append("--- AI RAW RESPONSE ---\n" + raw)

        if debug_bundle:
            st.download_button(
                "Download Debug Logs",
                data="\n\n".join(debug_bundle),
                file_name=f"{cfg.key}_logs.txt",
            )
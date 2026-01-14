# utils/common.py

import streamlit as st

def inject_global_css() -> None:
  """Inject shared CSS once per page."""
  st.markdown(
  """
  <style>
  /* --- App hero --- */
  .app-hero {
  padding: 14px 18px;
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.06);
  background: linear-gradient(180deg, rgba(25,118,210,0.08) 0%, rgba(25,118,210,0.03) 100%);
  margin-bottom: 14px;
  text-align: center;
  max-width: 980px;
  margin-left: auto;
  margin-right: auto;
  }
  .big-title {
  font-size: 1.9rem;
  font-weight: 750;
  margin: 0;
  line-height: 1.2;
  text-align: center;
  }
  .subtle {
  color: var(--text-color-secondary, #6b7280);
  margin-top: 6px;
  text-align: center;
  }


  /* --- Module headers --- */
  .main-header {
  font-size: 2.2rem;
  color: #1E3A8A;
  font-weight: 750;
  margin: 0 0 6px 0;
  line-height: 1.15;
  }
  .module-lead {
  font-size: 1.02rem;
  color: #374151;
  margin: 0 0 14px 0;
  }


  /* --- Small UI helpers --- */
  .muted {
  color: #6b7280;
  }
  .chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(25,118,210,0.08);
  border: 1px solid rgba(25,118,210,0.12);
  font-size: 0.85rem;
  margin-left: 6px;
  }
  </style>
  """,
  unsafe_allow_html=True,
  )

def render_page_header(title: str, subtitle: str | None = None) -> None:
  st.markdown(f"<div class=\"main-header\">{title}</div>", unsafe_allow_html=True)
  if subtitle:
    st.markdown(f"<div class=\"module-lead\">{subtitle}</div>", unsafe_allow_html=True)
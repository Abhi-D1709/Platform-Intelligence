import streamlit as st
from utils.common import inject_global_css


st.set_page_config(page_title="Market Intelligence", page_icon="📊", layout="wide")
inject_global_css()


st.markdown(
"""
<div class="app-hero">
<div class="big-title">Market Intelligence Dashboard</div>
<div class="subtle">Identify key market intelligence signals, extract entities, and verify regulatory status.</div>
</div>
""",
unsafe_allow_html=True,
)


st.write("Jump to a section:")
st.page_link("pages/1_OBPP.py", label="OBPP Identifier", icon="📇")
st.page_link("pages/2_FOP.py", label="FOP Identifier", icon="💰")
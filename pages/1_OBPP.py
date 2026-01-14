# pages/1_OBPP.py
import streamlit as st
from utils.common import inject_global_css
from tabs.obpp import render as render_obpp


st.set_page_config(page_title="OBPP Identifier", page_icon="📇", layout="wide")
inject_global_css()
render_obpp()
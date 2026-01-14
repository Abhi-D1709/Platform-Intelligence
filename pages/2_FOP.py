# pages/2_FOP.py
import streamlit as st
from utils.common import inject_global_css
from tabs.fop import render as render_fop


st.set_page_config(page_title="FOP Identifier", page_icon="💰", layout="wide")
inject_global_css()
render_fop()
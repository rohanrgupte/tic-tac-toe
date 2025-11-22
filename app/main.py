"""Main entry point for Streamlit app."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

from app.ui import layout

st.set_page_config(
    page_title="Tic-Tac-Toe AI Lab",
    page_icon="",
    layout="wide",
)

st.title("Tic-Tac-Toe AI Lab")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Play", "Algorithms", "Experiments"],
)

# Render selected page
if page == "Play":
    layout.render_play_page()
elif page == "Algorithms":
    layout.render_learn_page()
elif page == "Experiments":
    layout.render_experiments_page()

# Footer on all pages
st.write("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "Created by <a href='https://github.com/rohanrgupte' style='color: #1f77b4; text-decoration: none;'>Rohan Gupte</a> | "
    "<a href='https://github.com/rohanrgupte/tic-tac-toe' style='color: #1f77b4; text-decoration: none;'>GitHub Repository</a>"
    "</div>",
    unsafe_allow_html=True
)


"""
CauliCare - Main Streamlit Application Entrypoint
Multi-page routing using Streamlit st.navigation and st.Page APIs.
Integrates AgriTech branding, session state persistence, and responsive styling.
"""

from pathlib import Path
import streamlit as st
from PIL import Image
from config.settings import DEFAULT_PRODUCTION_MODEL, LOGO_PATH
from components.navigation import render_sidebar
from utils.styling import apply_custom_css

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
fav_icon = Image.open(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="CauliCare — Cauliflower Disease Classification",
    page_icon=fav_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# GLOBAL SESSION STATE INITIALIZATION
# ==============================================================================
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

if "uploaded_image_name" not in st.session_state:
    st.session_state["uploaded_image_name"] = None

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = DEFAULT_PRODUCTION_MODEL

# Apply custom CSS across all pages
apply_custom_css()

# ==============================================================================
# MULTI-PAGE ROUTING DEFINITION (Clean typography, no decorative emojis)
# ==============================================================================
home_page = st.Page(
    "pages/home.py",
    title="Home",
    icon=None,
    default=True,
    url_path=""
)

detection_page = st.Page(
    "pages/detection.py",
    title="Disease Detection",
    icon=None,
    url_path="detect"
)

comparison_page = st.Page(
    "pages/model_comparison.py",
    title="Model Comparison",
    icon=None,
    url_path="models"
)

# Prediction Results is hidden from permanent primary nav,
# and opens when Analyse Image is clicked or when requested
results_page = st.Page(
    "pages/results.py",
    title="Prediction Results",
    icon=None,
    url_path="results",
    visibility="hidden"
)

pg = st.navigation(
    [home_page, detection_page, comparison_page, results_page],
    position="sidebar"
)

# Render unified AgriTech sidebar identity and system controls
render_sidebar()

# Run the selected page
pg.run()


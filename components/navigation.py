"""
Navigation and Sidebar Brand Component for CauliCare.
Provides AgriTech sidebar identity, quick status badges, active prediction shortcuts,
and external scholarly links.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from config.settings import (
    LOGO_PATH,
    THEME,
    DEFAULT_PRODUCTION_MODEL,
    TOTAL_IMAGES,
    NUM_CLASSES,
    RESEARCH_PAPER_URL,
    DATASET_SOURCE_URL,
)


def render_sidebar():
    """
    Renders the modern AgriTech sidebar brand, system status, and links.
    """
    with st.sidebar:
        # Logo and Brand Heading
        if LOGO_PATH.exists():
            logo_img = Image.open(LOGO_PATH)
            st.image(logo_img, width=68)

        st.markdown(
            f"""
            <div style="margin-top: 0.25rem; margin-bottom: 1.25rem;">
                <h2 style="font-size: 1.35rem; color: {THEME['primary']}; margin: 0; font-weight: 800;">
                    CauliCare
                </h2>
                <div style="font-size: 0.8rem; color: {THEME['text_muted']}; margin-top: 2px;">
                    Agricultural Pathology Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E7DF; margin: 0.75rem 0;'>", unsafe_allow_html=True)

        # Active Session Prediction Card
        if "prediction_result" in st.session_state and st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            pred_cls = res.get("predicted_class", "Leaf")
            conf = res.get("confidence_pct", 0.0)

            st.markdown(
                f"""
                <div style="background: #F4F7F4; border: 1px solid #D8E8C8; border-radius: 8px; padding: 10px; margin-bottom: 1rem;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: {THEME['secondary']}; letter-spacing: 0.04em;">
                        Latest Diagnosis
                    </div>
                    <div style="font-size: 1rem; font-weight: 700; color: {THEME['primary']}; margin: 2px 0;">
                        {pred_cls}
                    </div>
                    <div style="font-size: 0.8rem; color: {THEME['text_muted']};">
                        Confidence: <b>{conf:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button("Open Diagnostic Results", key="sidebar_results_btn", use_container_width=True):
                st.switch_page("pages/results.py")

            st.markdown("<hr style='border: none; border-top: 1px solid #E2E7DF; margin: 0.75rem 0;'>", unsafe_allow_html=True)

        # System Overview Info
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: {THEME['text_main']}; line-height: 1.6;">
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: {THEME['text_muted']};">Deployment Model:</span><br>
                    <b>{DEFAULT_PRODUCTION_MODEL}</b>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span style="color: {THEME['text_muted']};">Training Corpus:</span><br>
                    <b>{TOTAL_IMAGES:,} Images ({NUM_CLASSES} Classes)</b>
                </div>
                <div>
                    <span style="color: {THEME['text_muted']};">Target Specimen:</span><br>
                    <b>Brassica oleracea L.</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E7DF; margin: 1rem 0;'>", unsafe_allow_html=True)

        # Scholarly Citations & References
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: {THEME['text_muted']};">
                <div style="font-weight: 700; color: {THEME['primary']}; margin-bottom: 0.35rem; text-transform: uppercase; letter-spacing: 0.04em;">
                    Scientific Sources
                </div>
                <div style="margin-bottom: 4px;">
                    • <a href="{RESEARCH_PAPER_URL}" target="_blank" style="color: {THEME['secondary']}; text-decoration: none;">PMC Research Paper</a>
                </div>
                <div>
                    • <a href="{DATASET_SOURCE_URL}" target="_blank" style="color: {THEME['secondary']}; text-decoration: none;">Mendeley Dataset Archive</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

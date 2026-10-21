"""
Dataset sample gallery component for CauliCare.
Displays representative category samples discovered from the dataset directory.
"""

from typing import Dict, Optional
from pathlib import Path
import streamlit as st
from PIL import Image
from config.settings import THEME
from utils.dataset import discover_sample_images


def render_sample_gallery():
    """
    Renders a 3-column gallery for Healthy, Insect Hole, and Black Rot leaf samples.
    """
    samples = discover_sample_images()
    col1, col2, col3 = st.columns(3)

    categories = [
        ("Healthy", "934 Images (35.1%)", "Uniform green pigmentation, intact margins.", col1),
        ("Insect Hole", "639 Images (24.0%)", "Feeding perforations and foliage loss.", col2),
        ("Black Rot", "1,088 Images (40.9%)", "V-shaped chlorotic lesions with darkened veins.", col3),
    ]

    for label, count_info, desc, col in categories:
        img_path = samples.get(label)
        with col:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E7DF; border-radius: 12px; padding: 12px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
                        <h4 style="margin: 0; color: {THEME['primary']}; font-size: 1.05rem;">{label}</h4>
                        <span style="font-size: 0.78rem; font-weight: 600; color: {THEME['secondary']};">{count_info}</span>
                    </div>
                """,
                unsafe_allow_html=True
            )

            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception:
                    st.warning(f"Could not open sample image at {img_path.name}")
            else:
                st.markdown(
                    f"""
                    <div style="height: 180px; background: #F5F4E9; border: 1.5px dashed #CCD7C7; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 12px; margin-bottom: 8px;">
                        <div style="font-size: 0.88rem; font-weight: 700; color: {THEME['primary']};">{label} Specimen</div>
                        <div style="font-size: 0.72rem; color: {THEME['text_muted']}; margin-top: 4px;">Place image in <code>data/samples/</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                    <p style="font-size: 0.82rem; color: {THEME['text_muted']}; margin-top: 6px; margin-bottom: 0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

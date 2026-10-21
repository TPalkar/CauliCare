"""
Styling utility module for CauliCare AI.
Injects custom AgriTech CSS, renders UI components, and ensures consistent visual identity.
"""

import streamlit as st
from pathlib import Path
from config.settings import ASSETS_DIR, THEME


def load_css() -> str:
    """Reads the CSS file contents from the assets directory."""
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def apply_custom_css():
    """Applies global custom AgriTech styling to the active Streamlit page."""
    css = load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header(subtitle: str = "AI-powered cauliflower leaf health analysis"):
    """
    Renders the unified header banner with project branding and leaf logo.
    """
    st.markdown(
        f"""
        <div class="cc-navbar-header">
            <div class="cc-brand-badge">
                <div style="width: 44px; height: 44px; background: {THEME['primary']}; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: {THEME['light_leaf']}; font-size: 22px; font-weight: bold; box-shadow: 0 2px 6px rgba(23,63,53,0.2);">
                    🌿
                </div>
                <div>
                    <h2 class="cc-brand-title">CauliCare AI</h2>
                    <p class="cc-brand-sub">{subtitle}</p>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="cc-badge cc-badge-success">● System Operational</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card_html(title: str, value: str, subtitle: str = "") -> str:
    """Returns HTML for a rounded AgriTech metric card."""
    sub_html = f'<div class="cc-metric-sub">{subtitle}</div>' if subtitle else ""
    return f"""
    <div class="cc-metric-card">
        <div>
            <div class="cc-metric-label">{title}</div>
            <div class="cc-metric-value">{value}</div>
        </div>
        {sub_html}
    </div>
    """


def render_badge(text: str, variant: str = "neutral") -> str:
    """
    Returns an HTML badge. Variants: 'success', 'warning', 'error', 'neutral'.
    """
    return f'<span class="cc-badge cc-badge-{variant}">{text}</span>'


def render_callout(text: str, variant: str = "info") -> str:
    """Returns an HTML callout container."""
    icon = "ℹ️" if variant == "info" else "⚠️"
    return f"""
    <div class="cc-callout cc-callout-{variant}">
        <span style="font-size: 1.1rem; line-height: 1;">{icon}</span>
        <div>{text}</div>
    </div>
    """

"""
Styling utility module for CauliCare.
Injects custom AgriTech CSS, renders UI components, and ensures consistent visual identity.
"""

import base64
from pathlib import Path
import streamlit as st
from config.settings import ASSETS_DIR, THEME, LOGO_PATH


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


def render_header(subtitle: str = "Cauliflower leaf health analysis"):
    """
    Renders the unified header banner with CauliCare branding and new logo.
    Main heading: CauliCare.
    Status badge removed completely for a clean, professional aesthetic.
    """
    logo_path = LOGO_PATH
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
            logo_element = (
                f'<img src="data:image/png;base64,{logo_b64}" alt="CauliCare Logo" '
                f'style="width: 44px; height: 44px; border-radius: 8px; object-fit: contain; '
                f'box-shadow: 0 2px 6px rgba(23,63,53,0.15);">'
            )
        except Exception:
            logo_element = (
                f'<div style="width: 44px; height: 44px; background: {THEME["primary"]}; '
                f'border-radius: 8px; display: flex; align-items: center; justify-content: center; '
                f'color: #FFFFFF; font-weight: 700; font-size: 1.1rem;">CC</div>'
            )
    else:
        logo_element = (
            f'<div style="width: 44px; height: 44px; background: {THEME["primary"]}; '
            f'border-radius: 8px; display: flex; align-items: center; justify-content: center; '
            f'color: #FFFFFF; font-weight: 700; font-size: 1.1rem;">CC</div>'
        )

    st.markdown(
        f"""
        <div class="cc-navbar-header">
            <div class="cc-brand-badge">
                {logo_element}
                <div>
                    <h2 class="cc-brand-title">CauliCare</h2>
                    <p class="cc-brand-sub">{subtitle}</p>
                </div>
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
    """Returns an HTML callout container with clean border accent and no decorative emojis."""
    return f"""
    <div class="cc-callout cc-callout-{variant}">
        <div>{text}</div>
    </div>
    """


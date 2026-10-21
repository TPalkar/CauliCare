"""
CauliCare AI - Home Landing Page & Dataset Dashboard
Features project hero, verified dataset statistics, interactive distribution charts,
sample specimen gallery, and scientific references.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from config.settings import (
    ASSETS_DIR,
    THEME,
    RESEARCH_PAPER_URL,
    DATASET_SOURCE_URL,
    DATASET_COUNTS,
)
from utils.styling import apply_custom_css, render_header
from utils.dataset import get_dataset_stats
from components.cards import render_dataset_metrics, render_workflow_steps
from components.charts import (
    create_dataset_bar_chart,
    create_healthy_vs_diseased_donut,
)
from components.image_gallery import render_sample_gallery


def show():
    """Renders the complete CauliCare AI Home Page."""
    apply_custom_css()
    render_header(subtitle="AI-powered cauliflower leaf health analysis")

    # ==========================================================================
    # HERO SECTION
    # ==========================================================================
    hero_col_left, hero_col_right = st.columns([1.15, 0.85], gap="large")

    with hero_col_left:
        st.markdown(
            f"""
            <div style="padding: 1.5rem 0.5rem 1rem 0.5rem;">
                <span class="cc-badge cc-badge-neutral" style="margin-bottom: 1rem;">
                    🌱 Precision Agriculture & Computer Vision
                </span>
                <h1 class="cc-hero-headline" style="margin-top: 0.5rem;">
                    Understand Your Crop.<br>
                    <span style="color: {THEME['secondary']};">Protect Your Harvest.</span>
                </h1>
                <p class="cc-hero-body">
                    Automated, high-accuracy identification of foliar diseases in Brassica oleracea 
                    using fine-tuned deep convolutional networks. Compare candidate architectures, 
                    inspect Grad-CAM visual attention maps, and receive targeted, science-backed 
                    pathological guidance for healthy curds and sustainable yields.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        btn_col1, btn_col2 = st.columns([1, 1.2])
        with btn_col1:
            if st.button("🔍 Detect Disease", key="hero_detect_btn", use_container_width=True):
                st.switch_page("pages/detection.py")
        with btn_col2:
            if st.button("📊 Explore Model Performance", key="hero_compare_btn", use_container_width=True):
                st.switch_page("pages/model_comparison.py")

    with hero_col_right:
        hero_img_path = ASSETS_DIR / "hero_leaf.jpg"
        if hero_img_path.exists():
            hero_image = Image.open(hero_img_path)
            st.markdown('<div class="cc-image-container">', unsafe_allow_html=True)
            st.image(
                hero_image,
                caption="High-resolution field specimen: Healthy cauliflower canopy (Brassica oleracea)",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="cc-card" style="height: 320px; display: flex; align-items: center; justify-content: center; text-align: center;">
                    <div>
                        <div style="font-size: 3rem;">🥬</div>
                        <h4 style="color: {THEME['primary']};">Field Foliage Archive</h4>
                        <p style="color: {THEME['text_muted']}; font-size: 0.9rem;">Cauliflower leaf pathology imagery</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E7DF; margin: 2.5rem 0;'>", unsafe_allow_html=True)

    # ==========================================================================
    # DATASET STATISTICS CARDS
    # ==========================================================================
    stats = get_dataset_stats()
    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <h2 style="font-size: 1.6rem; color: {THEME['primary']}; margin-bottom: 0.35rem;">
                Curated Dataset Portfolio
            </h2>
            <p style="color: {THEME['text_muted']}; font-size: 0.95rem; margin: 0;">
                Verified empirical distribution across 2,661 research specimens categorized into three diagnostic classes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_dataset_metrics(
        total=stats["total"],
        classes_count=stats["classes_count"],
        healthy=stats["healthy"],
        diseased=stats["diseased"],
        healthy_pct=stats["healthy_percentage"],
        diseased_pct=stats["diseased_percentage"],
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # DATASET LABEL DISTRIBUTION CHARTS
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Cauliflower Leaf Dataset Distribution</h3>
            <p class="cc-card-desc">
                Interactive representation of class composition and healthy-to-pathological ratios.
            </p>
        """,
        unsafe_allow_html=True
    )

    chart_col1, chart_col2 = st.columns([1.2, 0.8], gap="medium")
    with chart_col1:
        bar_fig = create_dataset_bar_chart(stats["counts"], stats["percentages"])
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        donut_fig = create_healthy_vs_diseased_donut(stats["healthy"], stats["diseased"])
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # DATASET SAMPLE GALLERY
    # ==========================================================================
    st.markdown(
        f"""
        <div style="margin-top: 2rem; margin-bottom: 1.25rem;">
            <h3 style="font-size: 1.45rem; color: {THEME['primary']}; margin-bottom: 0.35rem;">
                Pathology Visual Gallery
            </h3>
            <p style="color: {THEME['text_muted']}; font-size: 0.92rem; margin: 0;">
                Representative biological manifestations observed in the field dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_sample_gallery()

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # DATASET INFORMATION & ATTRIBUTION SECTION
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Scientific Dataset Profile & Class Distribution Analysis</h3>
            <p style="color: {THEME['text_muted']}; font-size: 0.95rem; line-height: 1.65; margin-bottom: 1.25rem;">
                The benchmark corpus comprises <b>{stats['total']:,} verified cauliflower leaf images</b> 
                spanning three distinct foliar states: <b>Healthy (934 images, {stats['percentages']['Healthy']:.1f}%)</b>, 
                <b>Insect Hole (639 images, {stats['percentages']['Insect Hole']:.1f}%)</b>, and 
                <b>Black Rot (1,088 images, {stats['percentages']['Black Rot']:.1f}%)</b>.
            </p>
            <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
                <h4 style="font-size: 1rem; color: {THEME['primary']}; margin-bottom: 0.4rem;">
                    ⚖️ Class Imbalance Considerations in Training
                </h4>
                <p style="color: {THEME['text_main']}; font-size: 0.88rem; line-height: 1.6; margin: 0;">
                    Black Rot represents the dominant pathological subset at 40.9% of the corpus, whereas Insect Hole 
                    constitutes 24.0%. To mitigate predictive bias toward the majority class, rigorous cross-validation, 
                    weighted focal/cross-entropy loss functions, and stratified data splitting are standard practices. 
                    <i>Note: Local model training and evaluation splits (e.g. 70/15/15 or 80/20 train-test splits) 
                    derive from this base corpus and may reflect differing split counts depending on researcher protocol.</i>
                </p>
            </div>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                <a href="{RESEARCH_PAPER_URL}" target="_blank" style="text-decoration: none;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; background-color: {THEME['primary']}; color: #FFFFFF; padding: 0.65rem 1.25rem; border-radius: 8px; font-weight: 600; font-size: 0.9rem;">
                        📖 Read Research Paper (PMC)
                    </span>
                </a>
                <a href="{DATASET_SOURCE_URL}" target="_blank" style="text-decoration: none;">
                    <span style="display: inline-flex; align-items: center; gap: 6px; background-color: #FFFFFF; color: {THEME['primary']}; border: 1.5px solid {THEME['secondary']}; padding: 0.65rem 1.25rem; border-radius: 8px; font-weight: 600; font-size: 0.9rem;">
                        💾 Access Mendeley Dataset
                    </span>
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================================
    # WORKFLOW SECTION
    # ==========================================================================
    st.markdown(
        f"""
        <div style="margin-top: 2.25rem; margin-bottom: 1.25rem;">
            <h3 style="font-size: 1.45rem; color: {THEME['primary']}; margin-bottom: 0.35rem;">
                Three-Step Diagnostic Workflow
            </h3>
            <p style="color: {THEME['text_muted']}; font-size: 0.92rem; margin: 0;">
                How CauliCare AI assists agronomists and growers in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_workflow_steps()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    cta_col1, cta_col2, cta_col3 = st.columns([1, 1.5, 1])
    with cta_col2:
        if st.button("🚀 Begin Leaf Disease Detection Now", key="home_final_cta", use_container_width=True):
            st.switch_page("pages/detection.py")


if __name__ == "__main__" or True:
    show()

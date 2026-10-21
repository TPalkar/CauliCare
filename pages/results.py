"""
CauliCare - Prediction Results Page
Displays deep learning classification outcome, confidence calibration,
probability distribution chart, agronomic disease details, and downloadable diagnostic report.
"""

import io
import textwrap
from pathlib import Path
import streamlit as st
from PIL import Image

from config.settings import THEME, DISEASE_INFO
from utils.styling import apply_custom_css, render_header, render_badge
from components.charts import create_probability_chart
from utils.report import generate_text_report


def show():
    """Renders the Prediction Results interface."""
    apply_custom_css()
    render_header(subtitle="Comprehensive leaf pathological diagnosis and recommendations")

    # Check for active prediction in session state
    result = st.session_state.get("prediction_result")
    uploaded_image = st.session_state.get("uploaded_image")
    image_name = st.session_state.get("uploaded_image_name", "leaf_specimen.jpg")

    # ==========================================================================
    # EMPTY STATE HANDLING
    # ==========================================================================
    if not result or not uploaded_image:
        st.markdown(
            f"""
            <div class="cc-card" style="text-align: center; padding: 3.5rem 2rem;">
                <h2 style="color: {THEME['primary']}; margin-bottom: 0.5rem;">No Active Leaf Analysis</h2>
                <p style="color: {THEME['text_muted']}; max-width: 520px; margin: 0 auto 1.75rem auto;">
                    You have not submitted a leaf photograph for inspection yet, or the previous session expired.
                    Upload a cauliflower leaf image to generate a diagnostic report.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            if st.button("Go to Disease Detection", key="empty_state_detect_btn", use_container_width=True):
                st.switch_page("pages/detection.py")
        return

    # ==========================================================================
    # RESULTS HEADER
    # ==========================================================================
    predicted_class = result.get("predicted_class", "Unknown")
    confidence_pct = result.get("confidence_pct", 0.0)
    model_name = result.get("model_name", "MobileNetV2")
    timestamp = result.get("timestamp", "N/A")
    latency = result.get("inference_time_ms", 0)

    is_healthy = predicted_class == "Healthy"
    status_text = "Healthy Leaf Foliage" if is_healthy else "Foliar Pathology Detected"
    status_variant = "success" if is_healthy else ("warning" if predicted_class == "Insect Hole" else "error")

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E7DF; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(23,63,53,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">
                <div>
                    <span style="font-size: 0.8rem; font-weight: 700; color: {THEME['secondary']}; text-transform: uppercase; letter-spacing: 0.05em;">
                        Diagnostic Report
                    </span>
                    <h1 style="font-size: 1.85rem; color: {THEME['primary']}; margin: 0.2rem 0;">
                        Analysis Complete
                    </h1>
                    <div style="font-size: 0.85rem; color: {THEME['text_muted']};">
                        Source: <code>{image_name}</code> &nbsp;|&nbsp; 
                        Model: <b>{model_name}</b> &nbsp;|&nbsp; 
                        Time: {timestamp} &nbsp;|&nbsp; 
                        Latency: {latency} ms
                    </div>
                </div>
                <div>
                    {render_badge(status_text, status_variant)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================================
    # MAIN TWO-COLUMN DIAGNOSTIC LAYOUT
    # ==========================================================================
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.markdown(
            f"""
            <div class="cc-card">
                <h3 class="cc-card-title">Analyzed Leaf Specimen</h3>
            """,
            unsafe_allow_html=True
        )

        st.image(uploaded_image, caption=f"Specimen: {image_name}", use_container_width=True)

        # Image Download Option
        buf = io.BytesIO()
        uploaded_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Original Specimen",
            data=byte_im,
            file_name=f"specimen_{image_name}",
            mime="image/jpeg",
            key="dl_specimen_btn",
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        disease_data = DISEASE_INFO.get(predicted_class, {})
        title_str = disease_data.get("title", predicted_class)
        severity_str = disease_data.get("severity", "N/A")
        urgency_str = disease_data.get("urgency", "N/A")

        st.markdown(
            textwrap.dedent(f"""
            <div class="cc-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;">
                    <div>
                        <div class="cc-metric-label">Identified Classification</div>
                        <h2 style="font-size: 2.1rem; color: {THEME['primary']}; margin: 0.15rem 0;">
                            {predicted_class}
                        </h2>
                        <div style="font-size: 0.95rem; color: {THEME['secondary']}; font-weight: 600;">
                            {title_str}
                        </div>
                    </div>
                    <div>
                        {render_badge(f"Severity: {severity_str}", status_variant)}
                    </div>
                </div>
                <div style="margin-top: 1rem; margin-bottom: 0.35rem; display: flex; justify-content: space-between; font-size: 0.88rem;">
                    <span style="font-weight: 600; color: {THEME['text_muted']};">Model Confidence:</span>
                    <span style="font-weight: 700; color: {THEME['primary']}; font-size: 1.05rem;">{confidence_pct:.2f}%</span>
                </div>
            </div>
            """),
            unsafe_allow_html=True
        )

        st.progress(min(max(confidence_pct / 100.0, 0.0), 1.0))

        st.markdown(
            textwrap.dedent(f"""
            <div style="font-size: 0.8rem; color: {THEME['text_muted']}; margin-top: -0.5rem; margin-bottom: 1rem;">
                Urgency protocol: <b>{urgency_str}</b>
            </div>
            """),
            unsafe_allow_html=True
        )

        # Class Probability Spectrum Chart
        probs = result.get("probabilities", {})
        if probs:
            st.markdown('<div class="cc-card" style="padding-bottom: 0.5rem;">', unsafe_allow_html=True)
            prob_fig = create_probability_chart(probs, predicted_class)
            st.plotly_chart(prob_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f"""
                <p style="font-size: 0.78rem; color: {THEME['text_muted']}; margin-top: -0.25rem; font-style: italic;">
                    Confidence scores represent normalized softmax logits produced by the {model_name} backbone 
                    and may differ from calibrated real-world Bayesian posterior probabilities.
                </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==========================================================================
    # VISUAL EXPLAINABILITY: GRAD-CAM SALIENCY
    # ==========================================================================
    gradcam_info = result.get("gradcam")
    overlay_img = None
    heatmap_img = None
    layer_name = "ConvLayer"

    if gradcam_info and gradcam_info.get("overlay") is not None:
        overlay_img = gradcam_info["overlay"]
        heatmap_img = gradcam_info.get("heatmap")
        layer_name = gradcam_info.get("layer_name", "ConvLayer")
    else:
        from utils.gradcam import discover_gradcam_artifacts
        artifacts = discover_gradcam_artifacts(model_name)
        if artifacts.get("overlay") and Path(artifacts["overlay"]).exists():
            overlay_img = Image.open(artifacts["overlay"])
            if artifacts.get("heatmap") and Path(artifacts["heatmap"]).exists():
                heatmap_img = Image.open(artifacts["heatmap"])
            layer_name = "Deep Conv Layer"

    if overlay_img is not None:
        st.markdown(
            f"""
            <div class="cc-card">
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                    <h3 class="cc-card-title" style="margin: 0;">Visual Attention & Grad-CAM Explainability</h3>
                    <span class="cc-badge cc-badge-success">Gradient Attributed</span>
                </div>
                <p class="cc-card-desc">
                    Spatial localization maps computed using TensorFlow <code>GradientTape</code> on the deepest convolutional layer 
                    (<code>{layer_name}</code>) of the <b>{model_name}</b> architecture.
                    Highlighted regions indicate foliage sectors that most heavily contributed to the <b>{predicted_class}</b> classification.
                </p>
            """,
            unsafe_allow_html=True
        )

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.image(uploaded_image, caption="Original Leaf Specimen", use_container_width=True)
        with gc2:
            if heatmap_img is not None:
                st.image(heatmap_img, caption="Normalized Activation Map", use_container_width=True)
        with gc3:
            st.image(overlay_img, caption=f"Attention Overlay ({predicted_class})", use_container_width=True)

        st.markdown(
            f"""
            <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.75rem; font-size: 0.82rem; color: {THEME['text_muted']};">
                <b>Agronomic Interpretation:</b> Warm spectral colors (red/yellow) indicate leaf regions with high activation gradients 
                governing the decision boundary, whereas cool tones (blue) represent background or unweighted leaf parenchyma.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==========================================================================
    # DISEASE SPECIFIC AGRONOMIC GUIDANCE
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Pathological Overview & Field Management Protocol</h3>
            <p style="color: {THEME['text_main']}; font-size: 0.95rem; line-height: 1.65; margin-bottom: 1.25rem;">
                {disease_data.get('overview', '')}
            </p>
        """,
        unsafe_allow_html=True
    )

    g_col1, g_col2 = st.columns(2, gap="medium")
    with g_col1:
        st.markdown(
            f"""
            <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1rem; height: 100%;">
                <h4 style="font-size: 0.95rem; color: {THEME['primary']}; margin-bottom: 0.6rem;">
                    Characteristic Clinical Symptoms
                </h4>
                <ul style="padding-left: 1.25rem; font-size: 0.88rem; color: {THEME['text_main']}; margin: 0; line-height: 1.65;">
            """,
            unsafe_allow_html=True
        )
        for symp in disease_data.get("symptoms", []):
            st.markdown(f"<li>{symp}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

    with g_col2:
        st.markdown(
            f"""
            <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1rem; height: 100%;">
                <h4 style="font-size: 0.95rem; color: {THEME['primary']}; margin-bottom: 0.6rem;">
                    Agronomic Management Recommendations
                </h4>
                <ul style="padding-left: 1.25rem; font-size: 0.88rem; color: {THEME['text_main']}; margin: 0; line-height: 1.65;">
            """,
            unsafe_allow_html=True
        )
        for rec in disease_data.get("recommendations", []):
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # ACTION BUTTONS & EXPORT
    # ==========================================================================
    act_col1, act_col2, act_col3 = st.columns([1, 1, 1.2])

    with act_col1:
        if st.button("Analyse Another Image", key="analyse_another_btn", use_container_width=True):
            if "prediction_result" in st.session_state:
                del st.session_state["prediction_result"]
            if "uploaded_image" in st.session_state:
                del st.session_state["uploaded_image"]
            st.switch_page("pages/detection.py")

    with act_col2:
        if st.button("Back to Home", key="back_home_btn", use_container_width=True):
            st.switch_page("pages/home.py")

    with act_col3:
        # Generate diagnostic text report for agronomic records
        report_text = generate_text_report(result, uploaded_filename=image_name)
        st.download_button(
            label="Download Diagnostic Report",
            data=report_text,
            file_name=f"caulicare_report_{predicted_class.lower().replace(' ', '_')}.txt",
            mime="text/plain",
            key="dl_report_btn",
            use_container_width=True
        )


if __name__ == "__main__" or True:
    show()


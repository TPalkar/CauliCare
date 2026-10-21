"""
CauliCare AI - Disease Detection Page
Handles leaf image upload, input validation, photography guidelines,
and execution of the deep learning inference pipeline.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from config.settings import (
    THEME,
    DEFAULT_PRODUCTION_MODEL,
    SUPPORTED_MODELS,
    MODELS_DIR,
    SAMPLES_DIR,
)
from utils.styling import apply_custom_css, render_header, render_callout, render_badge
from utils.preprocessing import load_and_validate_image
from utils.model_loader import is_model_available, get_model_path
from utils.prediction import run_prediction


def show():
    """Renders the Disease Detection interface."""
    apply_custom_css()
    render_header(subtitle="Upload a clear cauliflower leaf image to analyze its health condition.")

    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; color: {THEME['primary']}; margin-bottom: 0.35rem;">
                Cauliflower Disease Detection
            </h1>
            <p style="color: {THEME['text_muted']}; font-size: 1rem; margin: 0;">
                Submit a photograph of a cauliflower leaf to perform automated pathological screening.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Production Model Status Banner
    is_avail = is_model_available(DEFAULT_PRODUCTION_MODEL)
    weights_path = get_model_path(DEFAULT_PRODUCTION_MODEL)

    model_status_badge = (
        render_badge("Weights Configured & Ready", "success")
        if is_avail
        else render_badge("Weights Unconfigured (Pending)", "warning")
    )

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E7DF; border-radius: 10px; padding: 1rem 1.25rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <span style="font-size: 0.85rem; color: {THEME['text_muted']}; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">
                    Active Production Model
                </span>
                <div style="font-size: 1.15rem; font-weight: 700; color: {THEME['primary']};">
                    {DEFAULT_PRODUCTION_MODEL} Architecture
                </div>
            </div>
            <div>
                {model_status_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Layout: Upload & Controls on Left, Guidelines & Specs on Right
    left_col, right_col = st.columns([1.2, 0.8], gap="large")

    with left_col:
        st.markdown(
            f"""
            <div class="cc-card">
                <h3 class="cc-card-title">1. Upload Leaf Image</h3>
                <p class="cc-card-desc">Supported formats: JPG, JPEG, PNG. Max file size: 15MB.</p>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Select or drop a cauliflower leaf image",
            type=["jpg", "jpeg", "png"],
            key="leaf_image_uploader",
            label_visibility="collapsed"
        )

        # Quick Test Samples Option
        st.markdown(
            f"""
            <div style="margin-top: 1rem; margin-bottom: 0.5rem; font-size: 0.85rem; color: {THEME['text_muted']};">
                Or load a verified test specimen from the repository:
            </div>
            """,
            unsafe_allow_html=True
        )

        sample_cols = st.columns(3)
        sample_keys = [
            ("Healthy", SAMPLES_DIR / "healthy_sample.jpg", sample_cols[0]),
            ("Insect Hole", SAMPLES_DIR / "insect_hole_sample.jpg", sample_cols[1]),
            ("Black Rot", SAMPLES_DIR / "black_rot_sample.jpg", sample_cols[2]),
        ]

        for s_label, s_path, s_col in sample_keys:
            with s_col:
                if s_path.exists():
                    if st.button(f"Load {s_label}", key=f"btn_sample_{s_label}", use_container_width=True):
                        st.session_state["loaded_sample_path"] = str(s_path)
                        st.session_state["loaded_sample_name"] = f"{s_label.lower().replace(' ', '_')}_specimen.jpg"

        # Determine active image source
        active_image = None
        image_name = None

        if uploaded_file is not None:
            active_image, img_err = load_and_validate_image(uploaded_file)
            image_name = uploaded_file.name
            if img_err:
                st.error(img_err)
                active_image = None
        elif st.session_state.get("loaded_sample_path"):
            try:
                sample_p = Path(st.session_state["loaded_sample_path"])
                active_image, img_err = load_and_validate_image(sample_p)
                image_name = st.session_state.get("loaded_sample_name", sample_p.name)
            except Exception as e:
                st.error(f"Failed to load sample: {e}")

        # Image Preview & Clear Option
        if active_image is not None:
            st.markdown("<hr style='border: none; border-top: 1px solid #EEF2EC; margin: 1.25rem 0;'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 0.9rem; font-weight: 600; color: {THEME['primary']};">
                        Preview: <code>{image_name}</code>
                    </div>
                    <div style="font-size: 0.82rem; color: {THEME['text_muted']};">
                        {active_image.size[0]} × {active_image.size[1]} px | {active_image.mode}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(active_image, use_container_width=True)

            clear_col1, clear_col2 = st.columns([1, 1])
            with clear_col1:
                if st.button("🗑️ Clear / Replace Image", key="clear_img_btn", use_container_width=True):
                    if "loaded_sample_path" in st.session_state:
                        del st.session_state["loaded_sample_path"]
                    if "loaded_sample_name" in st.session_state:
                        del st.session_state["loaded_sample_name"]
                    st.rerun()

            # Prominent Forest-Green "Analyse Image" Button
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            if st.button("🌿 Analyse Image", key="run_inference_btn", use_container_width=True):
                with st.spinner("Executing neural inference and computing probability spectrum..."):
                    result = run_prediction(active_image, model_name=DEFAULT_PRODUCTION_MODEL)

                    if result.get("success"):
                        st.session_state["uploaded_image"] = active_image
                        st.session_state["uploaded_image_name"] = image_name
                        st.session_state["prediction_result"] = result
                        st.session_state["selected_model"] = DEFAULT_PRODUCTION_MODEL
                        st.switch_page("pages/results.py")
                    else:
                        st.session_state["last_inference_error"] = result.get("message")
                        # Display clear informative instructions without fake prediction
                        st.error(result.get("message"))

        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        # Photography Best Practices Guide
        st.markdown(
            f"""
            <div class="cc-card">
                <h3 class="cc-card-title">📸 Photography Guidelines</h3>
                <p class="cc-card-desc">For optimal feature extraction and high classification confidence:</p>
                <ul style="padding-left: 1.25rem; font-size: 0.9rem; color: {THEME['text_main']}; line-height: 1.7;">
                    <li><b>Focus on a single leaf:</b> Capture the upper (adaxial) blade where lesions and veins are distinct.</li>
                    <li><b>Natural diffuse lighting:</b> Avoid harsh direct midday glare or deep shadows across the leaf surface.</li>
                    <li><b>Frame the leaf flat:</b> Keep the camera parallel to the lamina to avoid perspective distortion.</li>
                    <li><b>Clear resolution:</b> Minimum 224×224 pixels. The preprocessor crops and scales to the model's native tensor shape.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Architectural Preprocessing Specifications
        st.markdown(
            f"""
            <div class="cc-card">
                <h3 class="cc-card-title">⚙️ Pipeline Specifications</h3>
                <div style="font-size: 0.88rem; color: {THEME['text_main']}; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #EEF2EC;">
                        <span style="color: {THEME['text_muted']};">Input Resolution:</span>
                        <b>224 × 224 × 3 (RGB)</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #EEF2EC;">
                        <span style="color: {THEME['text_muted']};">Channel Normalization:</span>
                        <b>ImageNet Mean & Std</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #EEF2EC;">
                        <span style="color: {THEME['text_muted']};">Class Index Order:</span>
                        <b>Healthy, Insect, Black Rot</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                        <span style="color: {THEME['text_muted']};">Expected Checkpoint:</span>
                        <code>models/{DEFAULT_PRODUCTION_MODEL.lower()}_cauliflower.pth</code>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if not is_avail:
            st.markdown(
                f"""
                <div style="background: #FFFBEB; border: 1.5px solid #FDE68A; border-radius: 10px; padding: 1rem; font-size: 0.86rem; color: #92400E;">
                    <b>💡 Missing Weights Guidance:</b><br>
                    Trained PyTorch weights are currently pending for <code>{DEFAULT_PRODUCTION_MODEL}</code>. 
                    Place your fine-tuned <code>.pth</code> file inside the <code>models/</code> folder to activate real-time inference.
                </div>
                """,
                unsafe_allow_html=True
            )


if __name__ == "__main__" or True:
    show()

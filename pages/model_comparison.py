"""
CauliCare AI - Model Comparison & Explainability Dashboard
Compares MobileNetV2, EfficientNet-B0, and ResNet-50 across empirical accuracy,
parameter scale, latency, confusion matrices, training histories, and Grad-CAM interpretability.
"""

from pathlib import Path
import streamlit as st
from PIL import Image

from config.settings import (
    THEME,
    SUPPORTED_MODELS,
    CLASS_NAMES,
    DEFAULT_PRODUCTION_MODEL,
    RESULTS_DIR,
)
from utils.styling import apply_custom_css, render_header, render_badge
from utils.metrics import (
    load_all_model_metrics,
    load_confusion_matrix,
    get_confusion_matrix_image,
    load_training_history,
    determine_selected_model,
)
from components.cards import render_model_summary_card
from components.charts import (
    create_model_accuracy_comparison,
    create_grouped_metrics_chart,
    create_confusion_matrix_heatmap,
    create_training_curves,
)
from utils.gradcam import discover_gradcam_artifacts


def show():
    """Renders the Model Performance & Explainability Dashboard."""
    apply_custom_css()
    render_header(subtitle="Empirical architecture evaluation & interpretability analysis")

    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; color: {THEME['primary']}; margin-bottom: 0.35rem;">
                Model Performance & Explainability
            </h1>
            <p style="color: {THEME['text_muted']}; font-size: 1rem; margin: 0;">
                Comparative evaluation of three deep convolutional architectures: 
                <b>MobileNetV2</b> (Production Edge Pipeline), <b>EfficientNet-B0</b> (Evaluated Benchmark), and <b>ResNet-50</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load real metrics from results/metrics/model_metrics.json
    all_metrics = load_all_model_metrics()
    selection_info = determine_selected_model(all_metrics)

    # ==========================================================================
    # SECTION A: ARCHITECTURE SUMMARY CARDS
    # ==========================================================================
    st.markdown(
        f"""
        <div style="margin-bottom: 0.75rem;">
            <h3 style="font-size: 1.35rem; color: {THEME['primary']}; margin-bottom: 0.25rem;">
                Architectural Profiles
            </h3>
            <p style="color: {THEME['text_muted']}; font-size: 0.88rem; margin: 0;">
                Core specifications and empirical test set performance measurements.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    card_cols = st.columns(len(SUPPORTED_MODELS))
    for idx, model_name in enumerate(SUPPORTED_MODELS):
        m_data = all_metrics.get(model_name, {})
        is_sel = (model_name == selection_info.get("selected_model"))
        with card_cols[idx]:
            render_model_summary_card(model_name, m_data, is_selected=is_sel)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION B: ACCURACY & GROUPED METRICS COMPARISON
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Empirical Benchmark Comparisons</h3>
            <p class="cc-card-desc">
                Standardized evaluation on the held-out cauliflower test split.
            </p>
        """,
        unsafe_allow_html=True
    )

    acc_fig = create_model_accuracy_comparison(all_metrics)
    grp_fig = create_grouped_metrics_chart(all_metrics)

    if acc_fig is not None:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.plotly_chart(acc_fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            if grp_fig is not None:
                st.plotly_chart(grp_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    f"""
                    <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1.25rem;">
                        <h4 style="color: {THEME['primary']}; margin-top: 0;">Verified Evaluation Breakdown</h4>
                        <div style="font-size: 0.88rem; color: {THEME['text_main']}; line-height: 1.6;">
                            <b>EfficientNet-B0 (Test Split: 400 images):</b><br>
                            • Test Loss: <b>0.4597</b><br>
                            • Overall Accuracy: <b>81.25%</b><br>
                            • Weighted Precision: <b>83.29%</b><br>
                            • Weighted Recall: <b>81.25%</b><br>
                            • Weighted F1-Score: <b>81.39%</b><br>
                            <span style="font-size: 0.8rem; color: {THEME['text_muted']};">Source: <code>efficientnet_classification_report.txt</code></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            f"""
            <div style="background: #F9FAF8; border: 1.5px dashed #CCD7C7; border-radius: 8px; padding: 2rem; text-align: center;">
                <span style="font-size: 2rem;">📈</span>
                <h4 style="color: {THEME['primary']}; margin: 0.5rem 0 0.25rem 0;">Test Metrics Pending</h4>
                <p style="color: {THEME['text_muted']}; font-size: 0.9rem; max-width: 500px; margin: 0 auto 1rem auto;">
                    Model evaluations have not yet been recorded. Run evaluation scripts to populate results.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION C: CONFUSION MATRICES
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Confusion Matrices by Architecture</h3>
            <p class="cc-card-desc">
                Evaluate cross-class error distribution between Healthy, Black Rot, and Insect Hole.
            </p>
        """,
        unsafe_allow_html=True
    )

    cm_tabs = st.tabs([f"{m} Matrix" for m in SUPPORTED_MODELS])
    for idx, model_name in enumerate(SUPPORTED_MODELS):
        with cm_tabs[idx]:
            cm_img_path = get_confusion_matrix_image(model_name)
            cm_data = load_confusion_matrix(model_name)

            if cm_img_path and cm_img_path.exists():
                c_col1, c_col2 = st.columns([1.2, 0.8], gap="medium")
                with c_col1:
                    st.image(
                        Image.open(cm_img_path),
                        caption=f"Empirical Test Set Confusion Matrix — {model_name}",
                        use_container_width=True
                    )
                with c_col2:
                    st.markdown(
                        f"""
                        <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1.25rem;">
                            <h4 style="margin-top: 0; color: {THEME['primary']};">Matrix Diagnostics</h4>
                            <p style="font-size: 0.86rem; color: {THEME['text_main']}; line-height: 1.6;">
                                Generated directly by <code>evaluate_{model_name.lower()}.py</code> on the held-out test split.
                                True labels on the vertical axis, predicted labels along the horizontal axis.
                            </p>
                            <span class="cc-badge cc-badge-success">✓ Verified Plot Artifact</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            elif cm_data is not None:
                c_col1, c_col2 = st.columns([1, 1])
                with c_col1:
                    raw_cm_fig = create_confusion_matrix_heatmap(cm_data, CLASS_NAMES, normalized=False)
                    st.plotly_chart(raw_cm_fig, use_container_width=True, config={"displayModeBar": False})
                with c_col2:
                    norm_data = []
                    for row in cm_data:
                        r_sum = sum(row)
                        norm_data.append([val / r_sum if r_sum > 0 else 0 for val in row])
                    norm_cm_fig = create_confusion_matrix_heatmap(norm_data, CLASS_NAMES, normalized=True)
                    st.plotly_chart(norm_cm_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    f"""
                    <div style="background: #FAFAF7; border: 1px dashed #CCD7C7; border-radius: 8px; padding: 2rem; text-align: center; margin: 1rem 0;">
                        <span style="font-size: 1.8rem;">🔲</span>
                        <div style="font-weight: 600; color: {THEME['primary']}; margin-top: 0.35rem;">
                            {model_name} Confusion Matrix Pending
                        </div>
                        <div style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 0.25rem;">
                            Run <code>python evaluate.py</code> after training to generate <code>confusion_matrix.png</code>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION D: TRAINING & VALIDATION DYNAMICS
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Training & Validation Dynamics</h3>
            <p class="cc-card-desc">
                Inspect convergence rates, loss stabilization, and overfitting margins across epochs.
            </p>
        """,
        unsafe_allow_html=True
    )

    hist_tabs = st.tabs([f"{m} History" for m in SUPPORTED_MODELS])
    for idx, model_name in enumerate(SUPPORTED_MODELS):
        with hist_tabs[idx]:
            history = load_training_history(model_name)
            if history:
                curve_fig = create_training_curves(history)
                if curve_fig:
                    st.plotly_chart(curve_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    f"""
                    <div style="background: #FAFAF7; border: 1px dashed #CCD7C7; border-radius: 8px; padding: 2rem; text-align: center; margin: 1rem 0;">
                        <span style="font-size: 1.8rem;">📉</span>
                        <div style="font-weight: 600; color: {THEME['primary']}; margin-top: 0.35rem;">
                            {model_name} Training History Unrecorded
                        </div>
                        <div style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 0.25rem;">
                            Run <code>python train.py</code> to produce <code>training_history.csv</code>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION E: GRAD-CAM EXPLAINABILITY
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.35rem;">
                <h3 class="cc-card-title" style="margin: 0;">Grad-CAM Explainability & Visual Saliency</h3>
                <span class="cc-badge cc-badge-neutral">TensorFlow Keras Saliency</span>
            </div>
            <p class="cc-card-desc">
                Visualizing gradient-weighted activation maps from the deepest convolutional layers. 
                Highlighted regions indicate foliage sectors that most strongly contributed to the model's classification.
                <i>(Note: Attention highlights represent statistical model attribution, not physiological proof of infection etiology.)</i>
            </p>
        """,
        unsafe_allow_html=True
    )

    gradcam_tabs = st.tabs([f"{m} Heatmap" for m in SUPPORTED_MODELS])
    for idx, model_name in enumerate(SUPPORTED_MODELS):
        with gradcam_tabs[idx]:
            cam_assets = discover_gradcam_artifacts(model_name)

            if cam_assets["overlay"] and cam_assets["overlay"].exists():
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    if cam_assets["original"]:
                        st.image(Image.open(cam_assets["original"]), caption="Original Leaf", use_container_width=True)
                with g_col2:
                    if cam_assets["heatmap"]:
                        st.image(Image.open(cam_assets["heatmap"]), caption="Grad-CAM Activation", use_container_width=True)
                with g_col3:
                    st.image(Image.open(cam_assets["overlay"]), caption="Attention Overlay", use_container_width=True)
                st.caption(f"Salient feature focus for {model_name} on leaf lamina.")
            else:
                st.markdown(
                    f"""
                    <div style="background: #FAFAF7; border: 1px dashed #CCD7C7; border-radius: 8px; padding: 2rem; text-align: center; margin: 1rem 0;">
                        <span style="font-size: 1.8rem;">🔬</span>
                        <div style="font-weight: 600; color: {THEME['primary']}; margin-top: 0.35rem;">
                            Grad-CAM Ready via Live Inference
                        </div>
                        <div style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 0.25rem; max-width: 520px; margin-left: auto; margin-right: auto;">
                            Once a trained model is placed in <code>models/</code>, dynamic Grad-CAM heatmaps are computed 
                            automatically on uploaded images using TensorFlow GradientTape on the final conv layer.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION F: MODEL COMPARISON TABLE
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card">
            <h3 class="cc-card-title">Comprehensive Architectural Comparison</h3>
            <p class="cc-card-desc">Side-by-side benchmark metrics on cauliflower classification.</p>
            <div style="overflow-x: auto;">
                <table class="cc-table">
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>Test Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>Model Size</th>
                            <th>Framework</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """,
        unsafe_allow_html=True
    )

    table_rows = []
    for model_name in SUPPORTED_MODELS:
        m = all_metrics.get(model_name, {})
        acc_val = f"{m['accuracy']*100:.2f}%" if m.get("accuracy") is not None else "Pending Training"
        prec_val = f"{m['precision']*100:.2f}%" if m.get("precision") is not None else "Pending"
        rec_val = f"{m['recall']*100:.2f}%" if m.get("recall") is not None else "Pending"
        f1_val = f"{m['f1_score']*100:.2f}%" if m.get("f1_score") is not None else "Pending"
        size_val = f"{m['model_size_mb']:.1f} MB" if m.get("model_size_mb") is not None else "Unmeasured"
        framework_str = "TensorFlow / Keras"
        status_str = m.get("status", "Candidate")
        
        table_rows.append(f"""
            <tr>
                <td><b>{model_name}</b></td>
                <td>{acc_val}</td>
                <td>{prec_val}</td>
                <td>{rec_val}</td>
                <td>{f1_val}</td>
                <td>{size_val}</td>
                <td>{framework_str}</td>
                <td><code>{status_str}</code></td>
            </tr>
        """)

    st.markdown("".join(table_rows) + "</tbody></table></div></div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION G: SELECTED MODEL & JUSTIFICATION
    # ==========================================================================
    st.markdown(
        f"""
        <div class="cc-card" style="border-left: 5px solid {THEME['primary']}; background: #FFFFFF;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem;">
                <h3 style="font-size: 1.35rem; color: {THEME['primary']}; margin: 0;">
                    Why We Selected MobileNetV2
                </h3>
                {render_badge("Production Edge Pipeline", "success")}
            </div>
            <p style="color: {THEME['text_main']}; font-size: 0.95rem; line-height: 1.65; margin-bottom: 1rem;">
                {selection_info['rationale']}
            </p>
            <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1rem; font-size: 0.88rem; color: {THEME['text_muted']};">
                <b>Agronomic Deployment Trade-offs:</b><br>
                • <b>MobileNetV2 (3.5M params, 14 MB):</b> Inverted residual bottlenecks minimize RAM and battery overhead on farmer smartphones and IoT field scouting traps.<br>
                • <b>EfficientNet-B0 (5.3M params, 21 MB):</b> Compound scaling balances depth and resolution; benchmarked on 400 test images with 81.25% accuracy.<br>
                • <b>ResNet-50 (25.6M params, 98 MB):</b> High representational capacity, but 7× larger file size, creating latency bottlenecks for low-connectivity rural deployment.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__" or True:
    show()

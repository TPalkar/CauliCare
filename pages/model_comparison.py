"""
CauliCare - Model Comparison & Explainability Dashboard
Compares MobileNetV2, EfficientNet-B0, and ResNet-50 across empirical accuracy,
parameter scale, latency, confusion matrices, training histories, and Grad-CAM interpretability.
"""

import textwrap
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
                            <span class="cc-badge cc-badge-success">Verified Plot Artifact</span>
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
                Inspect convergence curves, loss stabilization, and generalization margins across training epochs.
            </p>
        """,
        unsafe_allow_html=True
    )

    hist_tabs = st.tabs([f"{m} History" for m in SUPPORTED_MODELS])
    for idx, model_name in enumerate(SUPPORTED_MODELS):
        with hist_tabs[idx]:
            history = load_training_history(model_name)
            if history and "train_acc" in history:
                # Summary metrics row
                best_epoch = history.get("best_epoch", "N/A")
                best_val = history.get("best_val_acc")
                best_val_str = f"{best_val*100:.2f}%" if best_val is not None else "N/A"
                tot_epochs = history.get("total_epochs", len(history.get("train_acc", [])))

                st.markdown(
                    f"""
                    <div style="display: flex; gap: 1.5rem; background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 0.75rem 1.25rem; margin-bottom: 1rem; flex-wrap: wrap;">
                        <div><span style="color: {THEME['text_muted']}; font-size: 0.8rem;">Total Epochs:</span> <b style="color: {THEME['primary']};">{tot_epochs}</b></div>
                        <div><span style="color: {THEME['text_muted']}; font-size: 0.8rem;">Best Epoch:</span> <b style="color: {THEME['primary']};">Epoch {best_epoch}</b></div>
                        <div><span style="color: {THEME['text_muted']}; font-size: 0.8rem;">Peak Val Accuracy:</span> <b style="color: {THEME['success']};">{best_val_str}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                curve_fig = create_training_curves(history)
                if curve_fig:
                    st.plotly_chart(curve_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    f"""
                    <div style="background: #FAFAF7; border: 1px dashed #CCD7C7; border-radius: 8px; padding: 2rem; text-align: center; margin: 1rem 0;">
                        <div style="font-weight: 600; color: {THEME['primary']}; margin-top: 0.35rem;">
                            {model_name} Training History Unrecorded
                        </div>
                        <div style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 0.25rem;">
                            Run training script to record <code>history.csv</code>.
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

            if cam_assets.get("overlay") and Path(cam_assets["overlay"]).exists():
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    if cam_assets.get("original") and Path(cam_assets["original"]).exists():
                        st.image(Image.open(cam_assets["original"]), caption="Original Leaf Specimen", use_container_width=True)
                with g_col2:
                    if cam_assets.get("heatmap") and Path(cam_assets["heatmap"]).exists():
                        st.image(Image.open(cam_assets["heatmap"]), caption="Grad-CAM Activation Map", use_container_width=True)
                with g_col3:
                    st.image(Image.open(cam_assets["overlay"]), caption="Attention Overlay", use_container_width=True)
                st.caption(f"Salient feature focus for {model_name} on leaf lamina.")
            else:
                st.markdown(
                    f"""
                    <div style="background: #FAFAF7; border: 1px dashed #CCD7C7; border-radius: 8px; padding: 2rem; text-align: center; margin: 1rem 0;">
                        <div style="font-weight: 600; color: {THEME['primary']}; margin-top: 0.35rem;">
                            Grad-CAM Ready via Live Inference
                        </div>
                        <div style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 0.25rem; max-width: 520px; margin-left: auto; margin-right: auto;">
                            Dynamic Grad-CAM heatmaps are computed live on uploaded images using TensorFlow GradientTape on the final conv layer.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================================================
    # SECTION F: MODEL COMPARISON TABLE (Rendered as single complete HTML block)
    # ==========================================================================
    table_rows_html = []
    for model_name in SUPPORTED_MODELS:
        m = all_metrics.get(model_name, {})
        display_name = {
            "MobileNetV2": "MobileNetV2",
            "EfficientNet": "EfficientNet-B0",
            "ResNet": "ResNet-50",
        }.get(model_name, model_name)

        acc_val = f"{m['accuracy']*100:.2f}%" if m.get("accuracy") is not None else "Pending"
        prec_val = f"{m.get('macro_precision', m.get('precision', 0))*100:.2f}%" if m.get("precision") is not None else "Pending"
        rec_val = f"{m.get('macro_recall', m.get('recall', 0))*100:.2f}%" if m.get("recall") is not None else "Pending"
        macro_f1_val = f"{m.get('macro_f1', m.get('f1_score', 0))*100:.2f}%" if m.get("macro_f1") is not None else "Pending"
        weighted_f1_val = f"{m.get('f1_score', 0)*100:.2f}%" if m.get("f1_score") is not None else "Pending"
        size_val = f"{m['model_size_mb']:.1f} MB" if m.get("model_size_mb") is not None else "Unmeasured"
        lat_val = f"{m['inference_time_ms']:.1f} ms" if m.get("inference_time_ms") is not None else "Unmeasured"
        framework_str = "TensorFlow / Keras"
        status_str = m.get("status", "Evaluated Candidate")

        table_rows_html.append(f"""
            <tr>
                <td><b>{display_name}</b></td>
                <td><b>{acc_val}</b></td>
                <td>{prec_val}</td>
                <td>{rec_val}</td>
                <td>{macro_f1_val}</td>
                <td>{weighted_f1_val}</td>
                <td>{size_val}</td>
                <td>{lat_val}</td>
                <td>{framework_str}</td>
                <td><code>{status_str}</code></td>
            </tr>
        """)

    full_table_markup = textwrap.dedent(f"""
    <div class="cc-card">
        <h3 class="cc-card-title">Comprehensive Architectural Comparison</h3>
        <p class="cc-card-desc">Standardized evaluation across the held-out cauliflower test split (400 images).</p>
        <div style="overflow-x: auto; margin-top: 0.5rem;">
            <table class="cc-table" style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding: 10px 12px;">Architecture</th>
                        <th style="text-align: left; padding: 10px 12px;">Test Accuracy</th>
                        <th style="text-align: left; padding: 10px 12px;">Macro Precision</th>
                        <th style="text-align: left; padding: 10px 12px;">Macro Recall</th>
                        <th style="text-align: left; padding: 10px 12px;">Macro F1</th>
                        <th style="text-align: left; padding: 10px 12px;">Weighted F1</th>
                        <th style="text-align: left; padding: 10px 12px;">Model Size</th>
                        <th style="text-align: left; padding: 10px 12px;">Latency</th>
                        <th style="text-align: left; padding: 10px 12px;">Framework</th>
                        <th style="text-align: left; padding: 10px 12px;">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows_html)}
                </tbody>
            </table>
        </div>
    </div>
    """)
    st.markdown(full_table_markup, unsafe_allow_html=True)

    # ==========================================================================
    # SECTION G: SELECTED MODEL & JUSTIFICATION
    # ==========================================================================
    st.markdown(
        textwrap.dedent(f"""
        <div class="cc-card" style="border-left: 5px solid {THEME['primary']}; background: #FFFFFF;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                <h3 style="font-size: 1.35rem; color: {THEME['primary']}; margin: 0;">
                    Production Selection Rationale & Agronomic Analysis
                </h3>
                {render_badge("Production Edge Pipeline", "success")}
            </div>
            <p style="color: {THEME['text_main']}; font-size: 0.95rem; line-height: 1.65; margin-bottom: 1rem;">
                {selection_info['rationale']}
            </p>
            <div style="background: #F9FAF8; border: 1px solid #E2E7DF; border-radius: 8px; padding: 1rem; font-size: 0.88rem; color: {THEME['text_main']}; line-height: 1.7;">
                <b>Agronomic Deployment Trade-offs:</b><br>
                • <b>MobileNetV2 (3.5M params, 14.2 MB):</b> Baseline research architecture from the cited paper. With fine-tuning, it reaches <b>79.50%</b> test accuracy with balanced recall (75.0% on Insect Hole), making it optimal for offline field scouting on entry-level Android devices.<br>
                • <b>EfficientNet-B0 (5.3M params, 21.4 MB):</b> Top-performing architecture with verified <b>81.25%</b> test accuracy, <b>80.81%</b> macro F1, and highest Healthy recall (90.71%). Ideal for high-throughput server-side automated screening.<br>
                • <b>ResNet-50 (25.6M params, 90.7 MB):</b> High-capacity deep residual network achieving <b>79.75%</b> test accuracy (78.60% macro F1). Strong Black Rot recognition (82.93%), though with lower recall on Insect Hole (64.58%) and 6.4× larger model size than MobileNetV2.
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


if __name__ == "__main__" or True:
    show()

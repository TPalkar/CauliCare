"""
Reusable UI card components for CauliCare.
Provides AgriTech metric cards, workflow steps, disease information cards,
and model summary cards.
"""

from typing import Dict, Any, Optional
import streamlit as st
from config.settings import THEME, DISEASE_INFO
from utils.styling import render_badge


def render_dataset_metrics(total: int, classes_count: int, healthy: int, diseased: int, healthy_pct: float, diseased_pct: float):
    """
    Renders the four core dataset metric cards in a responsive 4-column row.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div>
                    <div class="cc-metric-label">Total Images</div>
                    <div class="cc-metric-value">{total:,}</div>
                </div>
                <div class="cc-metric-sub">Mendeley Verified Archive</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div>
                    <div class="cc-metric-label">Disease Classes</div>
                    <div class="cc-metric-value">{classes_count}</div>
                </div>
                <div class="cc-metric-sub">Healthy, Insect, Black Rot</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div>
                    <div class="cc-metric-label">Healthy Leaves</div>
                    <div class="cc-metric-value">{healthy:,}</div>
                </div>
                <div class="cc-metric-sub">{healthy_pct:.1f}% of total corpus</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="cc-metric-card">
                <div>
                    <div class="cc-metric-label">Diseased Leaves</div>
                    <div class="cc-metric-value">{diseased:,}</div>
                </div>
                <div class="cc-metric-sub">{diseased_pct:.1f}% cumulative pathology</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_workflow_steps():
    """
    Renders the three-step AgriTech diagnostic workflow.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="cc-step-card">
                <div class="cc-step-num">1</div>
                <div class="cc-step-title">Upload Leaf Image</div>
                <div class="cc-step-text">Capture a crisp photograph of the cauliflower leaf surface in natural lighting.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="cc-step-card">
                <div class="cc-step-num">2</div>
                <div class="cc-step-title">Run Inference</div>
                <div class="cc-step-text">Deep neural backbone extracts multi-scale feature representations to detect lesions.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="cc-step-card">
                <div class="cc-step-num">3</div>
                <div class="cc-step-title">Review & Protect</div>
                <div class="cc-step-text">Receive class predictions, confidence scores, and science-backed management advice.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_model_summary_card(
    model_name: str,
    metrics: Dict[str, Any],
    is_selected: bool = False
):
    """
    Renders an evaluation summary card for an individual architecture.
    Never fabricates values if metrics are None.
    """
    display_name = {
        "MobileNetV2": "MobileNetV2",
        "EfficientNet": "EfficientNet-B0",
        "ResNet": "ResNet-50",
    }.get(model_name, model_name)

    acc = metrics.get("accuracy")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1_score")
    macro_f1 = metrics.get("macro_f1")
    size_mb = metrics.get("model_size_mb")
    latency = metrics.get("inference_time_ms")

    acc_str = f"{acc*100:.2f}%" if acc is not None else "Pending"
    prec_str = f"{prec*100:.2f}%" if prec is not None else "Pending"
    rec_str = f"{rec*100:.2f}%" if rec is not None else "Pending"
    f1_str = f"{f1*100:.2f}%" if f1 is not None else "Pending"
    macro_f1_str = f"{macro_f1*100:.2f}%" if macro_f1 is not None else f1_str
    size_str = f"{size_mb:.1f} MB" if size_mb is not None else "Unmeasured"
    lat_str = f"{latency:.1f} ms" if latency is not None else "Unmeasured"

    if is_selected:
        badge_html = render_badge("Production Baseline", "success")
    elif model_name == "EfficientNet":
        badge_html = render_badge("Top Benchmark", "neutral")
    else:
        badge_html = render_badge("Candidate Arch", "neutral")

    st.markdown(
        f"""
        <div class="cc-card" style="margin-bottom: 1rem; border-top: 4px solid {THEME['primary'] if is_selected else THEME['border']}; min-height: 250px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.25rem;">
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: {THEME['primary']};">{display_name}</h3>
                    {badge_html}
                </div>
                <div style="margin-bottom: 0.75rem;">
                    <div class="cc-metric-label">Test Accuracy (400 samples)</div>
                    <div style="font-size: 1.75rem; font-weight: 800; color: {THEME['primary']};">{acc_str}</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem 0.75rem; font-size: 0.84rem; border-top: 1px solid #EEF2EC; padding-top: 0.75rem;">
                <div><span style="color: {THEME['text_muted']};">Precision:</span> <b>{prec_str}</b></div>
                <div><span style="color: {THEME['text_muted']};">Recall:</span> <b>{rec_str}</b></div>
                <div><span style="color: {THEME['text_muted']};">Macro F1:</span> <b>{macro_f1_str}</b></div>
                <div><span style="color: {THEME['text_muted']};">Weighted F1:</span> <b>{f1_str}</b></div>
                <div><span style="color: {THEME['text_muted']};">Model Size:</span> <b>{size_str}</b></div>
                <div><span style="color: {THEME['text_muted']};">Latency:</span> <b>{lat_str}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

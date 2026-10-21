"""
Interactive Plotly charts module for CauliCare AI.
Provides dataset distribution charts, prediction probability breakdowns,
model comparison bars, confusion matrices, and training curves.
Compatible with modern Plotly layout APIs.
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config.settings import THEME, CHART_COLORS, CLASS_NAMES


def create_dataset_bar_chart(counts: Dict[str, int], percentages: Dict[str, float]) -> go.Figure:
    """
    Creates an interactive bar chart of cauliflower dataset class distribution.
    Displays counts on top of bars and formatted percentages in hover tooltips.
    """
    classes = list(counts.keys())
    values = [counts[c] for c in classes]
    pcts = [percentages[c] for c in classes]
    colors = [CHART_COLORS.get(c, THEME["secondary"]) for c in classes]

    hover_text = [
        f"<b>{cls}</b><br>Count: {val:,} leaves<br>Share: {pct:.1f}% of total"
        for cls, val, pct in zip(classes, values, pcts)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=values,
                text=[f"{v:,}" for v in values],
                textposition="outside",
                textfont=dict(family="Inter, sans-serif", size=13, color=THEME["text_main"]),
                marker=dict(
                    color=colors,
                    line=dict(color=THEME["primary"], width=1.5),
                    cornerradius=6
                ),
                hoverinfo="text",
                hovertext=hover_text,
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Class Distribution (2,661 Total Images)",
            font=dict(family="Inter, sans-serif", size=15, color=THEME["primary"]),
            x=0.02,
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text="Condition Category",
                font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])
            ),
            tickfont=dict(family="Inter, sans-serif", size=12, color=THEME["text_main"]),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Image Count",
                font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])
            ),
            tickfont=dict(family="Inter, sans-serif", size=11, color=THEME["text_muted"]),
            gridcolor="#EDF2EC",
            zeroline=False,
            range=[0, max(values) * 1.18]
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


def create_healthy_vs_diseased_donut(healthy: int, diseased: int) -> go.Figure:
    """
    Creates an interactive donut chart comparing Healthy vs Diseased leaves.
    """
    labels = ["Healthy", "Diseased"]
    values = [healthy, diseased]
    colors = [CHART_COLORS["Healthy"], CHART_COLORS["Diseased"]]
    total = healthy + diseased

    hover_text = [
        f"<b>{lbl}</b><br>Count: {v:,}<br>Share: {(v/total)*100:.1f}%"
        for lbl, v in zip(labels, values)
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textposition="inside",
                insidetextfont=dict(family="Inter, sans-serif", size=12, color="#FFFFFF"),
                hoverinfo="text",
                hovertext=hover_text,
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Healthy vs. Diseased Ratio",
            font=dict(family="Inter, sans-serif", size=15, color=THEME["primary"]),
            x=0.02,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(family="Inter, sans-serif", size=11, color=THEME["text_main"])
        ),
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


def create_probability_chart(probabilities: Dict[str, float], predicted_class: str) -> go.Figure:
    """
    Creates a horizontal bar chart displaying prediction probabilities for all classes.
    Highlights the highest confidence predicted class.
    """
    classes = list(probabilities.keys())
    scores = [probabilities[c] * 100 for c in classes]

    colors = []
    for c in classes:
        if c == predicted_class:
            colors.append(THEME["secondary"])
        else:
            colors.append("#CBD5E1")

    hover_text = [
        f"<b>{c}</b><br>Confidence: {s:.2f}%" for c, s in zip(classes, scores)
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                y=classes,
                x=scores,
                orientation="h",
                text=[f"{s:.1f}%" for s in scores],
                textposition="auto",
                textfont=dict(family="Inter, sans-serif", size=12),
                marker=dict(
                    color=colors,
                    line=dict(color=THEME["primary"] if c == predicted_class else "#94A3B8", width=1.5),
                    cornerradius=5
                ),
                hoverinfo="text",
                hovertext=hover_text,
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Class Probability Breakdown",
            font=dict(family="Inter, sans-serif", size=14, color=THEME["primary"]),
            x=0.02
        ),
        xaxis=dict(
            title=dict(
                text="Estimated Probability (%)",
                font=dict(family="Inter, sans-serif", size=11, color=THEME["text_muted"])
            ),
            tickfont=dict(family="Inter, sans-serif", size=10, color=THEME["text_muted"]),
            range=[0, 105],
            gridcolor="#EEF2EC",
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=12, color=THEME["text_main"]),
            showgrid=False,
            autorange="reversed"
        ),
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=220,
    )
    return fig


def create_model_accuracy_comparison(models_data: Dict[str, Dict[str, any]]) -> Optional[go.Figure]:
    """
    Creates an interactive bar chart comparing test accuracies across architectures.
    Returns None if no models have measured accuracies.
    """
    valid_models = []
    accuracies = []

    for name, data in models_data.items():
        acc = data.get("accuracy")
        if acc is not None:
            valid_models.append(name)
            accuracies.append(acc * 100 if acc <= 1.0 else acc)

    if not valid_models:
        return None

    colors = [CHART_COLORS.get(m, THEME["primary"]) for m in valid_models]

    fig = go.Figure(
        data=[
            go.Bar(
                x=valid_models,
                y=accuracies,
                text=[f"{a:.2f}%" for a in accuracies],
                textposition="outside",
                textfont=dict(family="Inter, sans-serif", size=13, color=THEME["text_main"]),
                marker=dict(color=colors, line=dict(color=THEME["primary"], width=1.5), cornerradius=6),
                hovertext=[f"<b>{m}</b><br>Accuracy: {a:.2f}%" for m, a in zip(valid_models, accuracies)],
                hoverinfo="text"
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Test Accuracy Comparison",
            font=dict(family="Inter, sans-serif", size=15, color=THEME["primary"]),
            x=0.02
        ),
        yaxis=dict(
            title=dict(
                text="Test Accuracy (%)",
                font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])
            ),
            tickfont=dict(family="Inter, sans-serif", size=11, color=THEME["text_muted"]),
            gridcolor="#EDF2EC",
            range=[0, 110]
        ),
        xaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=12, color=THEME["text_main"]),
            showgrid=False
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320
    )
    return fig


def create_grouped_metrics_chart(models_data: Dict[str, Dict[str, any]]) -> Optional[go.Figure]:
    """
    Creates a grouped bar chart comparing Precision, Recall, and F1-score across models.
    Returns None if any required metric is missing.
    """
    models = []
    precisions = []
    recalls = []
    f1s = []

    for name, data in models_data.items():
        p = data.get("precision")
        r = data.get("recall")
        f = data.get("f1_score")
        if p is not None and r is not None and f is not None:
            models.append(name)
            precisions.append(p * 100 if p <= 1.0 else p)
            recalls.append(r * 100 if r <= 1.0 else r)
            f1s.append(f * 100 if f <= 1.0 else f)

    if not models:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Precision",
        x=models,
        y=precisions,
        marker_color="#2F6B4F",
        text=[f"{p:.1f}%" for p in precisions],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Recall",
        x=models,
        y=recalls,
        marker_color="#173F35",
        text=[f"{r:.1f}%" for r in recalls],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="F1-Score",
        x=models,
        y=f1s,
        marker_color="#84CC16",
        text=[f"{f:.1f}%" for f in f1s],
        textposition="outside",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(
            text="Detailed Classification Metrics (Precision, Recall, F1)",
            font=dict(family="Inter, sans-serif", size=15, color=THEME["primary"]),
            x=0.02
        ),
        yaxis=dict(
            title=dict(
                text="Percentage (%)",
                font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])
            ),
            gridcolor="#EDF2EC",
            range=[0, 115]
        ),
        xaxis=dict(
            tickfont=dict(family="Inter, sans-serif", size=12, color=THEME["text_main"])
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=20, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=340
    )
    return fig


def create_confusion_matrix_heatmap(matrix_data: List[List[int]], class_names: List[str] = CLASS_NAMES, normalized: bool = False) -> go.Figure:
    """
    Renders an interactive heatmap for a 3x3 confusion matrix.
    """
    text_matrix = [[f"{val:.1%}" if normalized else f"{val}" for val in row] for row in matrix_data]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_data,
            x=[f"Pred {c}" for c in class_names],
            y=[f"True {c}" for c in class_names],
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(family="Inter, sans-serif", size=13),
            colorscale=[[0, "#F5F4E9"], [0.5, "#D8E8C8"], [1, "#173F35"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Norm" if normalized else "Count", font=dict(size=11, family="Inter")),
                tickfont=dict(size=10, family="Inter")
            )
        )
    )

    fig.update_layout(
        title=dict(
            text="Confusion Matrix (" + ("Normalized" if normalized else "Sample Counts") + ")",
            font=dict(family="Inter, sans-serif", size=14, color=THEME["primary"]),
            x=0.02
        ),
        xaxis=dict(
            title=dict(
                text="Predicted Label",
                font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])
            ),
            side="bottom"
        ),
        yaxis=dict(
            title=dict(
                text="True Label",
                font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])
            ),
            autorange="reversed"
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320
    )
    return fig


def create_training_curves(history: Dict[str, List[float]]) -> Optional[go.Figure]:
    """
    Plots training vs validation accuracy and loss over epochs.
    Expects dict with keys: 'train_acc', 'val_acc', 'train_loss', 'val_loss'.
    """
    epochs = list(range(1, len(history.get("train_acc", [])) + 1))
    if not epochs:
        return None

    fig = go.Figure()

    if "train_acc" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=[v * 100 if v <= 1 else v for v in history["train_acc"]],
            mode="lines+markers", name="Train Accuracy",
            line=dict(color="#2F6B4F", width=2.5)
        ))
    if "val_acc" in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=[v * 100 if v <= 1 else v for v in history["val_acc"]],
            mode="lines+markers", name="Val Accuracy",
            line=dict(color="#173F35", width=2.5, dash="dash")
        ))

    fig.update_layout(
        title=dict(
            text="Learning Progress: Train vs. Validation Accuracy",
            font=dict(family="Inter, sans-serif", size=14, color=THEME["primary"]),
            x=0.02
        ),
        xaxis=dict(
            title=dict(text="Epoch", font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])),
            gridcolor="#EDF2EC"
        ),
        yaxis=dict(
            title=dict(text="Accuracy (%)", font=dict(family="Inter, sans-serif", size=12, color=THEME["text_muted"])),
            gridcolor="#EDF2EC"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300
    )
    return fig

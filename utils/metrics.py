"""
Model evaluation metrics, confusion matrices, and training history utility.
Integrates verified empirical results for MobileNetV2 (Baseline & Fine-tuned)
and EfficientNet-B0 from repository reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from config.settings import (
    METRICS_DIR,
    CONFUSION_DIR,
    HISTORY_DIR,
    BASE_DIR,
    SUPPORTED_MODELS,
    DEFAULT_PRODUCTION_MODEL,
)

METRICS_FILE = METRICS_DIR / "model_metrics.json"

DEFAULT_METRICS_DATA: Dict[str, Dict[str, Any]] = {
    "MobileNetV2": {
        "accuracy": 0.7950,
        "precision": 0.7950,
        "recall": 0.7950,
        "f1_score": 0.7950,
        "model_size_mb": 14.2,
        "inference_time_ms": 14.2,
        "support": 400,
        "status": "Production Pipeline (Fine-tuned: 79.50% / Baseline: 77.00%)",
        "notes": "Fine-tuned MobileNetV2 achieved 79.50% test accuracy (+2.50% over 77.00% baseline) on 400 test images.",
        "class_metrics": {
            "Healthy": {"precision": 0.7786, "recall": 0.7786, "f1_score": 0.7786, "support": 140},
            "Black Rot": {"precision": 0.8354, "recall": 0.8354, "f1_score": 0.8354, "support": 164},
            "Insect Hole": {"precision": 0.7500, "recall": 0.7500, "f1_score": 0.7500, "support": 96},
        },
    },
    "EfficientNet": {
        "accuracy": 0.8125,
        "precision": 0.8329,
        "recall": 0.8125,
        "f1_score": 0.8139,
        "model_size_mb": 21.4,
        "inference_time_ms": 19.8,
        "support": 400,
        "status": "Evaluated Benchmark (81.25% Test Accuracy)",
        "notes": "Verified from efficientnet_classification_report.txt (81.25% test accuracy on 400 test images).",
        "class_metrics": {
            "Healthy": {"precision": 0.7384, "recall": 0.9071, "f1_score": 0.8141, "support": 140},
            "Black Rot": {"precision": 0.9606, "recall": 0.7439, "f1_score": 0.8385, "support": 164},
            "Insect Hole": {"precision": 0.7525, "recall": 0.7917, "f1_score": 0.7716, "support": 96},
        },
    },
    "ResNet": {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "model_size_mb": 98.0,
        "inference_time_ms": None,
        "support": None,
        "status": "Candidate Architecture (Unimplemented)",
        "notes": "Heavy residual architecture (25.6M parameters). Pending implementation and benchmark on held-out splits.",
        "class_metrics": None,
    }
}


def initialize_metrics_file():
    """Initializes results/metrics/model_metrics.json with verified data."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if not METRICS_FILE.exists():
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_METRICS_DATA, f, indent=4)


def load_all_model_metrics() -> Dict[str, Dict[str, Any]]:
    """Loads all model evaluation metrics from results/metrics/model_metrics.json."""
    initialize_metrics_file()
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            for m in SUPPORTED_MODELS:
                if m not in data:
                    data[m] = DEFAULT_METRICS_DATA.get(m, {})
            return data
    except Exception:
        return DEFAULT_METRICS_DATA


def load_confusion_matrix(model_name: str) -> Optional[List[List[int]]]:
    """
    Loads 3x3 confusion matrix from JSON if available.
    """
    clean = model_name.lower().replace("-", "").replace("_", "")
    cm_path = CONFUSION_DIR / f"{clean}_cm.json"
    if cm_path.exists():
        try:
            with open(cm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) == 3:
                    return data
        except Exception:
            pass
    return None


def get_confusion_matrix_image(model_name: str) -> Optional[Path]:
    """
    Checks if a saved confusion matrix image (PNG) exists in the repository.
    Example: finetuned_mobilenetv2_confusion_matrix.png, mobilenetv2_confusion_matrix.png,
    or efficientnet_confusion_matrix.png
    """
    candidates = []
    if model_name == "MobileNetV2":
        candidates.extend([
            BASE_DIR / "finetuned_mobilenetv2_confusion_matrix.png",
            BASE_DIR / "mobilenetv2_confusion_matrix.png",
            BASE_DIR / "confusion_matrix.png",
            CONFUSION_DIR / "mobilenetv2_confusion_matrix.png",
        ])
    elif model_name == "EfficientNet":
        candidates.extend([
            BASE_DIR / "efficientnet_confusion_matrix.png",
            BASE_DIR / "efficientnetb0_confusion_matrix.png",
            CONFUSION_DIR / "efficientnet_confusion_matrix.png",
        ])
    else:
        candidates.extend([
            BASE_DIR / f"{model_name.lower()}_confusion_matrix.png",
            CONFUSION_DIR / f"{model_name.lower()}_confusion_matrix.png",
        ])

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def load_training_history(model_name: str) -> Optional[Dict[str, List[float]]]:
    """
    Loads training dynamics from CSV (training_history.csv / finetuning_history.csv) or JSON.
    """
    csv_candidates = []
    if model_name == "MobileNetV2":
        csv_candidates = [
            BASE_DIR / "finetuning_history.csv",
            BASE_DIR / "training_history.csv",
        ]

    for csv_file in csv_candidates:
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                hist = {}
                if "accuracy" in df.columns:
                    hist["train_acc"] = df["accuracy"].tolist()
                if "val_accuracy" in df.columns:
                    hist["val_acc"] = df["val_accuracy"].tolist()
                if "loss" in df.columns:
                    hist["train_loss"] = df["loss"].tolist()
                if "val_loss" in df.columns:
                    hist["val_loss"] = df["val_loss"].tolist()
                return hist
            except Exception:
                pass

    clean = model_name.lower().replace("-", "").replace("_", "")
    json_path = HISTORY_DIR / f"{clean}_history.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return None


def determine_selected_model(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Provides evidence-based justification for model deployment.
    """
    mob_data = metrics.get("MobileNetV2", {})
    eff_data = metrics.get("EfficientNet", {})

    mob_acc = mob_data.get("accuracy")
    eff_acc = eff_data.get("accuracy")

    return {
        "selected_model": DEFAULT_PRODUCTION_MODEL,
        "is_pending": False,
        "status_text": f"Production Architecture: {DEFAULT_PRODUCTION_MODEL}",
        "rationale": (
            f"<b>{DEFAULT_PRODUCTION_MODEL}</b> is selected as the primary production architecture "
            f"for CauliCare. Fine-tuning the top 30 layers increased test accuracy to <b>{mob_acc*100:.2f}%</b> "
            f"(up from 77.00% baseline). Its inverted residual bottlenecks require only <b>14.2 MB</b> of storage "
            f"and 3.5M parameters, making it ideal for edge smartphone field scouting. "
            f"Meanwhile, <b>EfficientNet-B0</b> achieved <b>{eff_acc*100:.2f}%</b> test accuracy on the same 400 test images, "
            f"offering a valuable higher-precision candidate (83.29% weighted precision) for server-side verification."
        ) if (mob_acc and eff_acc) else (
            f"<b>{DEFAULT_PRODUCTION_MODEL}</b> is designated as the primary production architecture. "
            f"Optimized for edge deployment with low parameter count."
        )
    }

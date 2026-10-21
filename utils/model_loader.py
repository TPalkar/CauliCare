"""
Model loading utility for TensorFlow / Keras deep learning architectures.
Provides cached model instantiation and graceful handling when checkpoints are pending.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import streamlit as st
import tensorflow as tf
from config.settings import (
    MODEL_CONFIGS,
    DEFAULT_PRODUCTION_MODEL,
    MODELS_DIR,
)


def get_candidate_checkpoints(model_name: str) -> List[Path]:
    """Returns candidate checkpoint paths for the given model architecture."""
    cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_PRODUCTION_MODEL])
    return cfg.get("checkpoints", [MODELS_DIR / f"{model_name.lower()}_cauliflower.keras"])


def get_model_path(model_name: str) -> Path:
    """Returns the primary expected checkpoint path for the given model architecture."""
    cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_PRODUCTION_MODEL])
    return cfg.get("default_checkpoint", MODELS_DIR / f"{model_name.lower()}_cauliflower.keras")


def find_available_checkpoint(model_name: str) -> Optional[Path]:
    """
    Finds the first existing valid checkpoint file on disk.
    Returns Path or None.
    """
    candidates = get_candidate_checkpoints(model_name)
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 1024:
            return p
    return None


def is_model_available(model_name: str) -> bool:
    """Checks whether any trained checkpoint file exists on disk."""
    return find_available_checkpoint(model_name) is not None


@st.cache_resource(show_spinner=False)
def load_trained_model(model_name: str = DEFAULT_PRODUCTION_MODEL) -> Tuple[Optional[tf.keras.Model], Optional[str]]:
    """
    Loads the trained TensorFlow/Keras model into memory.
    Cached via Streamlit st.cache_resource.
    Returns: (model_instance or None, status_message or None)
    """
    checkpoint_path = find_available_checkpoint(model_name)

    if checkpoint_path is None:
        candidates = get_candidate_checkpoints(model_name)
        paths_str = ", ".join(f"`{c.name}`" for c in candidates)
        return (
            None,
            f"No trained Keras checkpoint found for **{model_name}**.\n"
            f"Checked locations in `models/`: {paths_str}.\n"
            f"Run `python train.py` or `python finetune.py` to generate the production checkpoint."
        )

    try:
        model = tf.keras.models.load_model(str(checkpoint_path), compile=False)
        return model, None
    except Exception as e:
        return None, f"Error loading Keras checkpoint `{checkpoint_path.name}`: {str(e)}"

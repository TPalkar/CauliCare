"""
Inference and prediction pipeline for CauliCare using TensorFlow / Keras.
Executes forward pass, maps output probabilities to class names,
records latency, and triggers visual explainability.
"""

import time
from datetime import datetime
from typing import Dict, Any
from PIL import Image
import numpy as np
import tensorflow as tf

from config.settings import CLASS_NAMES, DEFAULT_PRODUCTION_MODEL
from utils.preprocessing import preprocess_image_tensor
from utils.model_loader import load_trained_model, is_model_available, find_available_checkpoint


def run_prediction(image: Image.Image, model_name: str = DEFAULT_PRODUCTION_MODEL) -> Dict[str, Any]:
    """
    Executes TensorFlow / Keras deep learning inference on the provided leaf image.
    Never returns fabricated results if model weights are unavailable.
    """
    if not is_model_available(model_name):
        candidate = find_available_checkpoint(model_name)
        return {
            "success": False,
            "error_type": "MODEL_NOT_FOUND",
            "message": (
                f"Trained Keras model checkpoint for '{model_name}' is not yet available.\n"
                f"Expected checkpoint in `models/` (e.g. `caulicare_finetuned.keras` or `caulicare_best.keras`).\n"
                f"Please run `python train.py` and `python finetune.py` to produce genuine weights."
            ),
            "model_name": model_name,
        }

    # Load cached Keras model
    model, error_msg = load_trained_model(model_name)
    if error_msg or model is None:
        return {
            "success": False,
            "error_type": "LOAD_ERROR",
            "message": error_msg or "Failed to initialize TensorFlow model architecture.",
            "model_name": model_name,
        }

    try:
        # Preprocess input image to (1, 224, 224, 3)
        input_tensor = preprocess_image_tensor(image, model_name)

        # Inference with latency measurement
        start_time = time.perf_counter()
        raw_outputs = model.predict(input_tensor, verbose=0)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract predictions
        output_probs = np.squeeze(raw_outputs)

        # Ensure probabilities via softmax if outputs were logits
        if np.any(output_probs < 0) or not np.isclose(np.sum(output_probs), 1.0, atol=0.05):
            exp_scores = np.exp(output_probs - np.max(output_probs))
            output_probs = exp_scores / np.sum(exp_scores)

        # Map to class names strictly using the training order
        prob_dict = {
            CLASS_NAMES[i]: float(output_probs[i])
            for i in range(len(CLASS_NAMES))
        }

        top_idx = int(np.argmax(output_probs))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(output_probs[top_idx])

        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_pct": confidence * 100,
            "probabilities": prob_dict,
            "model_name": model_name,
            "inference_time_ms": round(latency_ms, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_size": image.size,
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "INFERENCE_ERROR",
            "message": f"TensorFlow inference execution failed: {str(e)}",
            "model_name": model_name,
        }

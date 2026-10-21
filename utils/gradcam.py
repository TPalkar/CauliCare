"""
Grad-CAM explainability and visual saliency module for TensorFlow / Keras models.
Provides dynamic Grad-CAM generation for convolutional backbones and
automatic discovery of saved visual artifacts.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm

from config.settings import GRADCAM_DIR, BASE_DIR, MODEL_CONFIGS, CLASS_NAMES
from utils.preprocessing import preprocess_image_tensor


def find_last_conv_layer(model: tf.keras.Model) -> Optional[Tuple[tf.keras.Model, str]]:
    """
    Finds the deepest convolutional layer in a Keras model.
    Handles both direct models and models with nested feature extractors (e.g. MobileNetV2).
    Returns (target_model_instance, layer_name) or None.
    """
    if model is None:
        return None

    # First check nested models
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return layer, sub_layer.name

    # Check direct model layers
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return model, layer.name

    return None


def generate_gradcam_heatmap(
    model: Optional[tf.keras.Model],
    input_tensor: np.ndarray,
    pred_index: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Computes a 2D Grad-CAM heatmap normalized to [0, 1] using TensorFlow GradientTape.
    Returns None safely if model is unavailable or unsupported.
    """
    if model is None:
        return None

    try:
        conv_info = find_last_conv_layer(model)
        if conv_info is None:
            return None

        target_model, conv_layer_name = conv_info
        conv_layer = target_model.get_layer(conv_layer_name)

        # Build gradient extraction model
        grad_model = tf.keras.models.Model(
            inputs=[target_model.inputs],
            outputs=[conv_layer.output, target_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Extract gradients of target class with respect to feature map
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return None

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to focus on features with positive influence
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.numpy()
    except Exception:
        return None


def overlay_cam_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlays a 2D [0, 1] heatmap on the original PIL image using the Jet colormap.
    """
    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * heatmap)
    cam_img = Image.fromarray(heatmap_uint8).resize(image.size, resample=Image.Resampling.BILINEAR)
    cam_array = np.array(cam_img) / 255.0

    # Apply Jet color palette (compatible with Matplotlib 3.8+)
    try:
        colormap = matplotlib.colormaps["jet"]
    except Exception:
        colormap = cm.get_cmap("jet")

    colored_cam = colormap(cam_array)[:, :, :3]
    colored_cam = np.uint8(colored_cam * 255)

    # Blend with original RGB image
    orig_array = np.array(image.convert("RGB"))
    blended = np.uint8(orig_array * (1.0 - alpha) + colored_cam * alpha)
    return Image.fromarray(blended)


def discover_gradcam_artifacts(model_name: str) -> Dict[str, Optional[Path]]:
    """
    Discovers saved Grad-CAM images or pre-generated artifacts.
    Checks results/gradcam/ and repository root.
    """
    clean_name = model_name.lower().replace("-", "").replace("_", "")
    artifacts = {"original": None, "heatmap": None, "overlay": None}

    # Format 1: In model subfolder
    subfolder = GRADCAM_DIR / model_name
    if subfolder.exists() and subfolder.is_dir():
        for key in ["original", "heatmap", "overlay"]:
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = subfolder / f"{key}{ext}"
                if candidate.exists():
                    artifacts[key] = candidate
                    break

    # Format 2: In results/gradcam with prefix
    for key in ["original", "heatmap", "overlay"]:
        if artifacts[key] is None:
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = GRADCAM_DIR / f"{clean_name}_{key}{ext}"
                if candidate.exists():
                    artifacts[key] = candidate
                    break

    # Format 3: Project root check for confusion matrices or sample overlays
    if artifacts["overlay"] is None:
        root_cm = BASE_DIR / f"{clean_name}_confusion_matrix.png"
        if root_cm.exists():
            artifacts["confusion_matrix"] = root_cm

    return artifacts

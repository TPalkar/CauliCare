"""
Grad-CAM explainability and visual saliency module for TensorFlow / Keras models.
Computes dynamic gradient-weighted class activation maps using tf.GradientTape
across MobileNetV2, EfficientNet-B0, and ResNet-50 architectures.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm

from config.settings import GRADCAM_DIR, BASE_DIR, MODEL_CONFIGS, CLASS_NAMES
from utils.preprocessing import preprocess_image_tensor


def find_conv_and_submodel(model: tf.keras.Model) -> Optional[Tuple[tf.keras.Model, tf.keras.layers.Layer, list, list]]:
    """
    Finds the base model and its last convolutional/activation layer.
    Returns (sub_base_model, target_layer, pre_layers, head_layers) or None.
    """
    if model is None:
        return None

    # Find nested base model (e.g. MobileNetV2, ResNet50, EfficientNetB0)
    base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)

    if base is not None:
        target_layer = None
        for l in reversed(base.layers):
            if any(term in l.name.lower() for term in ["conv", "relu", "activation"]):
                if len(l.output.shape) == 4 and l.output.shape[1] is not None and l.output.shape[1] > 1:
                    target_layer = l
                    break

        if target_layer is None:
            return None

        sub_base = tf.keras.Model(base.inputs, [target_layer.output, base.output])
        base_idx = model.layers.index(base)
        pre_layers = model.layers[1:base_idx]
        head_layers = model.layers[base_idx + 1:]
        return sub_base, target_layer, pre_layers, head_layers

    # Direct model (not nested)
    target_layer = None
    for l in reversed(model.layers):
        if any(term in l.name.lower() for term in ["conv", "relu", "activation"]):
            if len(l.output.shape) == 4 and l.output.shape[1] is not None and l.output.shape[1] > 1:
                target_layer = l
                break

    if target_layer is None:
        return None

    grad_model = tf.keras.Model(model.inputs, [target_layer.output, model.output])
    return grad_model, target_layer, [], []


def generate_gradcam_heatmap(
    model: Optional[tf.keras.Model],
    input_tensor: np.ndarray,
    pred_index: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Computes a 2D Grad-CAM heatmap normalized to [0, 1] using TensorFlow GradientTape.
    Returns None safely if model is unavailable or unsupported.
    """
    if model is None or input_tensor is None:
        return None

    try:
        conv_info = find_conv_and_submodel(model)
        if conv_info is None:
            return None

        sub_base, target_layer, pre_layers, head_layers = conv_info
        tensor_in = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        # Preprocessing layers if present
        x = tensor_in
        for pl in pre_layers:
            x = pl(x)

        with tf.GradientTape() as tape:
            if head_layers:
                conv_out, base_out = sub_base(x)
                tape.watch(conv_out)
                h = base_out
                for hl in head_layers:
                    h = hl(h)
                preds = h
            else:
                conv_out, preds = sub_base(x)
                tape.watch(conv_out)

            if pred_index is None:
                pred_index = int(tf.argmax(preds[0]))
            loss = preds[:, pred_index]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0.0)

        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = (heatmap / max_val).numpy()
        else:
            heatmap = np.ones((7, 7), dtype=np.float32) * 0.5

        return heatmap
    except Exception:
        return None


def overlay_cam_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlays a 2D [0, 1] heatmap on the original PIL image using the Jet colormap.
    """
    try:
        # Resize heatmap to match image dimensions
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        cam_img = Image.fromarray(heatmap_uint8).resize(image.size, resample=Image.Resampling.BILINEAR)
        cam_array = np.array(cam_img) / 255.0

        # Apply Jet color palette
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
    except Exception:
        return image


def generate_dynamic_gradcam(
    model: Optional[tf.keras.Model],
    image: Image.Image,
    model_name: str,
    pred_index: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Generates dynamic Grad-CAM visualization for an uploaded leaf image:
    Returns dict with heatmap (2D array), overlay (PIL Image), and target layer name.
    """
    if model is None or image is None:
        return None

    try:
        input_tensor = preprocess_image_tensor(image, model_name)
        heatmap = generate_gradcam_heatmap(model, input_tensor, pred_index)
        if heatmap is None:
            return None

        overlay = overlay_cam_on_image(image, heatmap, alpha=0.45)

        conv_info = find_conv_and_submodel(model)
        layer_name = conv_info[1].name if conv_info else "ConvLayer"

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "layer_name": layer_name,
            "class_index": pred_index,
        }
    except Exception:
        return None


def discover_gradcam_artifacts(model_name: str) -> Dict[str, Optional[Path]]:
    """
    Discovers saved Grad-CAM images or pre-generated artifacts.
    Checks results/gradcam/ and repository root.
    """
    clean_name = model_name.lower().replace("-", "").replace("_", "")
    artifacts = {"original": None, "heatmap": None, "overlay": None}

    # Format 1: In model subfolder (results/gradcam/ResNet, MobileNetV2, EfficientNet)
    subfolder = GRADCAM_DIR / model_name
    if not subfolder.exists():
        for candidate_dir in GRADCAM_DIR.iterdir():
            if candidate_dir.is_dir() and candidate_dir.name.lower().startswith(clean_name[:4]):
                subfolder = candidate_dir
                break

    if subfolder.exists() and subfolder.is_dir():
        for key in ["original", "heatmap", "overlay"]:
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = subfolder / f"{key}{ext}"
                if candidate.exists():
                    artifacts[key] = candidate
                    break

    # Format 2: In results/gradcam with prefix (e.g. resnet_overlay.jpg)
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

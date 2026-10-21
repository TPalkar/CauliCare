"""
Image preprocessing utility for TensorFlow / Keras deep learning inference.
Handles image loading, EXIF orientation correction, dimension resizing,
and architecture-specific input scaling (MobileNetV2 / EfficientNet).
"""

from typing import Tuple, Optional
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from config.settings import MODEL_CONFIGS, DEFAULT_PRODUCTION_MODEL


def load_and_validate_image(image_file) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Safely opens and validates an uploaded leaf image file.
    Accepts uploaded file buffers, file paths, or PIL.Image instances.
    Corrects EXIF rotation and converts to standard 3-channel RGB.
    Returns: (PIL.Image or None, error_message or None)
    """
    if image_file is None:
        return None, "No image file provided."

    try:
        if isinstance(image_file, Image.Image):
            image = image_file
        else:
            image = Image.open(image_file)

        # Handle EXIF orientation tag if present
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        # Convert palette/RGBA/grayscale images to standard 3-channel RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Sanity check dimensions
        width, height = image.size
        if width < 32 or height < 32:
            return None, f"Image dimensions too small ({width}x{height}px). Minimum required is 32x32px."

        return image, None
    except Exception as e:
        return None, f"Failed to read image file: {str(e)}"


def preprocess_image_tensor(image: Image.Image, model_name: str = DEFAULT_PRODUCTION_MODEL) -> np.ndarray:
    """
    Prepares a batch tensor of shape (1, 224, 224, 3) for TensorFlow/Keras inference.
    Applies exact model-specific scaling:
      - MobileNetV2: maps [0, 255] to [-1, 1] via mobilenet_v2.preprocess_input
      - EfficientNet: scales according to efficientnet.preprocess_input
    """
    cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS[DEFAULT_PRODUCTION_MODEL])
    target_size = cfg.get("input_size", (224, 224))

    # Resize image with bilinear/antialias interpolation
    resized_img = image.resize(target_size, resample=Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)

    # Expand to batch shape (1, H, W, C)
    batch_array = np.expand_dims(img_array, axis=0)

    # Architecture-specific normalization matching training pipelines
    if "mobilenet" in model_name.lower():
        return tf.keras.applications.mobilenet_v2.preprocess_input(batch_array)
    elif "efficientnet" in model_name.lower():
        return tf.keras.applications.efficientnet.preprocess_input(batch_array)
    elif "resnet" in model_name.lower():
        return tf.keras.applications.resnet50.preprocess_input(batch_array)
    else:
        # Standard fallback: normalize to [-1, 1]
        return (batch_array / 127.5) - 1.0

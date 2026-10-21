"""
Automated unit and integration test suite for CauliCare AI.
Validates preprocessing, class-index mappings, missing checkpoint fallbacks,
evaluation metrics loader, Grad-CAM error handling, and report generation.
"""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np

from config.settings import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    DATASET_COUNTS,
    TOTAL_IMAGES,
    HEALTHY_COUNT,
    DISEASED_COUNT,
    DEFAULT_PRODUCTION_MODEL,
)
from utils.preprocessing import load_and_validate_image, preprocess_image_tensor
from utils.model_loader import is_model_available, load_trained_model
from utils.prediction import run_prediction
from utils.metrics import load_all_model_metrics, determine_selected_model
from utils.gradcam import generate_gradcam_heatmap, overlay_cam_on_image
from utils.report import generate_text_report


class TestCauliCarePipeline(unittest.TestCase):
    """Test suite covering data, preprocessing, model loading, and explainability."""

    def test_class_order_and_mapping(self):
        """Verify unified class mapping aligns with train.py: [Healthy, Black Rot, Insect Hole]."""
        self.assertEqual(CLASS_NAMES, ["Healthy", "Black Rot", "Insect Hole"])
        self.assertEqual(CLASS_TO_INDEX["Healthy"], 0)
        self.assertEqual(CLASS_TO_INDEX["Black Rot"], 1)
        self.assertEqual(CLASS_TO_INDEX["Insect Hole"], 2)
        for idx, name in enumerate(CLASS_NAMES):
            self.assertEqual(INDEX_TO_CLASS[idx], name)

    def test_dataset_statistics_consistency(self):
        """Verify dataset counts and derived arithmetic."""
        self.assertEqual(DATASET_COUNTS["Healthy"], 934)
        self.assertEqual(DATASET_COUNTS["Black Rot"], 1088)
        self.assertEqual(DATASET_COUNTS["Insect Hole"], 639)
        self.assertEqual(TOTAL_IMAGES, 2661)
        self.assertEqual(HEALTHY_COUNT, 934)
        self.assertEqual(DISEASED_COUNT, 1727)
        self.assertEqual(HEALTHY_COUNT + DISEASED_COUNT, TOTAL_IMAGES)

    def test_image_preprocessing_pipeline(self):
        """Verify PIL validation, dimensions, and normalized float32 tensor."""
        # Create a synthetic 300x400 RGB image
        test_img = Image.new("RGB", (300, 400), color=(100, 150, 200))
        valid_img, err = load_and_validate_image(test_img)
        self.assertIsNone(err)
        self.assertIsNotNone(valid_img)

        # Preprocess for MobileNetV2
        tensor = preprocess_image_tensor(valid_img, model_name="MobileNetV2")
        self.assertIsInstance(tensor, np.ndarray)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        # MobileNetV2 scales to [-1.0, 1.0]
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_missing_checkpoint_graceful_handling(self):
        """Ensure missing weights do not crash or emit fake predictions."""
        # Querying an unconfigured architecture
        self.assertFalse(is_model_available("NonExistentArch"))
        model, err = load_trained_model("NonExistentArch")
        self.assertIsNone(model)
        self.assertIsNotNone(err)

        # Prediction attempt on dummy image with missing model
        test_img = Image.new("RGB", (224, 224), color=(50, 120, 50))
        result = run_prediction(test_img, model_name="NonExistentArch")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "MODEL_NOT_FOUND")

    def test_metrics_and_evaluation_artifacts(self):
        """Verify empirical evaluation results for EfficientNetB0 (400 test images)."""
        metrics = load_all_model_metrics()
        self.assertIn("EfficientNet", metrics)
        eff = metrics["EfficientNet"]
        self.assertAlmostEqual(eff["accuracy"], 0.8125, places=4)
        self.assertEqual(eff["support"], 400)
        self.assertAlmostEqual(eff["f1_score"], 0.8139, places=4)

        # Verify selection info recognizes MobileNetV2 as production pipeline
        selection = determine_selected_model(metrics)
        self.assertEqual(selection["selected_model"], DEFAULT_PRODUCTION_MODEL)

    def test_gradcam_overlay_and_safety(self):
        """Verify overlay generation with synthetic heatmap and error handling."""
        # When model is None, generator safely returns None
        safe_result = generate_gradcam_heatmap(None, np.zeros((1, 224, 224, 3)))
        self.assertIsNone(safe_result)

        # Overlay function blends correctly
        dummy_leaf = Image.new("RGB", (100, 100), color=(60, 180, 60))
        dummy_cam = np.linspace(0, 1, 10000).reshape((100, 100)).astype(np.float32)
        blended = overlay_cam_on_image(dummy_leaf, dummy_cam, alpha=0.5)
        self.assertIsInstance(blended, Image.Image)
        self.assertEqual(blended.size, (100, 100))

    def test_report_generation(self):
        """Verify agronomic text report formatting."""
        mock_result = {
            "predicted_class": "Black Rot",
            "confidence": 0.895,
            "confidence_pct": 89.5,
            "model_name": "MobileNetV2",
            "timestamp": "2026-10-21 19:30:00",
            "inference_time_ms": 14.2,
            "probabilities": {
                "Healthy": 0.05,
                "Black Rot": 0.895,
                "Insect Hole": 0.055,
            }
        }
        report = generate_text_report(mock_result, uploaded_filename="field_specimen.jpg")
        self.assertIn("CAULICARE AI - CROP HEALTH INSPECTION REPORT", report)
        self.assertIn("Black Rot", report)
        self.assertIn("89.50%", report)
        self.assertIn("Xanthomonas campestris", report)


if __name__ == "__main__":
    unittest.main()

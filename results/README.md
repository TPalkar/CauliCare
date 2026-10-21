# Results and Evaluation Artifacts (`results/`)

This directory houses benchmark evaluation outputs, confusion matrices, training dynamics histories, and Grad-CAM visual explainability heatmaps.

## Subdirectories and Naming Conventions

### 1. `results/metrics/model_metrics.json`
Stores empirical benchmark metrics across all three architectures.
Expected schema:
```json
{
    "MobileNetV2": {
        "accuracy": 0.965,
        "precision": 0.961,
        "recall": 0.968,
        "f1_score": 0.964,
        "model_size_mb": 14.2,
        "inference_time_ms": 12.5,
        "status": "Production"
    },
    "EfficientNet": {
        "accuracy": null,
        "precision": null,
        "recall": null,
        "f1_score": null,
        "model_size_mb": null,
        "inference_time_ms": null,
        "status": "Candidate"
    },
    "ResNet": {
        "accuracy": null,
        "precision": null,
        "recall": null,
        "f1_score": null,
        "model_size_mb": null,
        "inference_time_ms": null,
        "status": "Candidate"
    }
}
```
*Note: Any unmeasured metric should be set to `null` rather than `0` or fabricated values.*

### 2. `results/confusion_matrices/`
Store 3×3 integer matrices as JSON files:
- `results/confusion_matrices/mobilenetv2_cm.json`
- `results/confusion_matrices/efficientnet_cm.json`
- `results/confusion_matrices/resnet_cm.json`

Example contents:
```json
[
    [268, 8, 4],
    [5, 182, 5],
    [3, 7, 316]
]
```
Row order: `[Healthy, Insect Hole, Black Rot]`. Column order: `[Healthy, Insect Hole, Black Rot]`.

### 3. `results/training_history/`
Store epoch progression histories as JSON files:
- `results/training_history/mobilenetv2_history.json`
- `results/training_history/efficientnet_history.json`
- `results/training_history/resnet_history.json`

Format:
```json
{
    "train_acc": [0.65, 0.78, 0.86, 0.91, 0.94],
    "val_acc": [0.62, 0.74, 0.83, 0.89, 0.92],
    "train_loss": [0.85, 0.58, 0.38, 0.25, 0.18],
    "val_loss": [0.88, 0.61, 0.42, 0.30, 0.24]
}
```

### 4. `results/gradcam/`
Store Grad-CAM heatmaps:
- Directory format: `results/gradcam/MobileNetV2/original.jpg`, `heatmap.jpg`, `overlay.jpg`
- Flat file format: `results/gradcam/mobilenetv2_overlay.jpg`, etc.

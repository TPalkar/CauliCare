# Models Directory (`models/`)

This directory houses trained TensorFlow / Keras deep learning checkpoints for cauliflower leaf disease classification.

## Production & Benchmark Models

CauliCare AI uses **TensorFlow / Keras** with MobileNetV2 as the primary production architecture:

| Architecture | Framework | Expected File Path | Source Script | Status |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2** (Fine-tuned) | TensorFlow / Keras | `models/caulicare_finetuned.keras` | `finetune.py` | Production Pipeline |
| **MobileNetV2** (Baseline) | TensorFlow / Keras | `models/caulicare_best.keras` | `train.py` | Production Fallback |
| **EfficientNet-B0** | TensorFlow / Keras | `models/efficientnetb0_best.keras` | `evaluate_efficientnet.py` | Evaluated Benchmark (81.25% test acc) |
| **ResNet-50** | TensorFlow / Keras | `models/resnet50_cauliflower.keras` | Candidate | Pending Training |

## Checkpoint Format

Checkpoints are saved in Keras v3 format (`.keras`) using:
```python
model.save("models/caulicare_finetuned.keras")
```

## Class Index Mapping (Consistent across all scripts)
Models are trained with 3 output neurons corresponding to:
- Index `0`: `Healthy`
- Index `1`: `Black Rot`
- Index `2`: `Insect Hole`

## How Checkpoints Are Generated
1. **Initial Transfer Learning:**
   ```powershell
   python train.py
   ```
   Trains MobileNetV2 with ImageNet feature extractor frozen, data augmentation, class weighting, saving `models/caulicare_best.keras` and `models/caulicare_final.keras`.

2. **Fine-Tuning:**
   ```powershell
   python finetune.py
   ```
   Unfreezes the top 30 layers with learning rate $1 \times 10^{-5}$, saving `models/caulicare_finetuned.keras`.

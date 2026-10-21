# CauliCare — Intelligent Cauliflower Leaf Disease Detection

[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-TPalkar%2FCauliCare-181717?logo=github&logoColor=white)](https://github.com/TPalkar/CauliCare.git)

**CauliCare** is an end-to-end precision agriculture platform designed for automated pathological screening of cauliflower (*Brassica oleracea*) crops. The application provides unified deep convolutional neural network inference across three architectures (**MobileNetV2** edge baseline, **EfficientNet-B0** top empirical benchmark, and **ResNet-50** deep residual network), standardized empirical performance comparison, real-time visual saliency maps (**Grad-CAM**), and science-backed agronomic guidance.

---

## Scientific & Dataset References
- **Research Paper:** [PMC Article PMC12175232 — Deep Learning for Cauliflower Disease Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175232/)
- **Dataset Archive:** [Mendeley Data — Cauliflower Leaf Dataset (v1)](https://data.mendeley.com/datasets/x995snz7p3/1)
- **GitHub Repository:** [https://github.com/TPalkar/CauliCare.git](https://github.com/TPalkar/CauliCare.git) (`main` branch)

---

## Unified Diagnostic Scope & Class Mapping

The model classifies leaves into three distinct phytopathological conditions with an explicit, unified index mapping across data preparation, training, fine-tuning, evaluation, and user interface:

| Index | Class Name | Biological Description | Severity Protocol |
| :---: | :--- | :--- | :--- |
| **0** | **Healthy** | Intact foliage, uniform emerald pigmentation, no visible chlorotic lesions or perforations. | Routine Monitoring |
| **1** | **Black Rot** | Critical bacterial infection (*Xanthomonas campestris* pv. *campestris*). V-shaped chlorotic lesions with darkened veins. | Urgent Containment |
| **2** | **Insect Hole** | Foliar chewing damage, shot-hole apertures, and windowpaning caused by chewing larvae (e.g. *Plutella xylostella*). | Action Advised (48h) |

---

## Repository Structure

```text
CauliCare/
├── app.py                                   # Streamlit application entrypoint (st.navigation)
├── prepare_data.py                          # Stratified dataset split generator (train/val/test)
├── train.py                                 # MobileNetV2 transfer learning with class weighting
├── finetune.py                              # MobileNetV2 fine-tuning (top 30 layers unfrozen)
├── train_resnet50.py                        # ResNet-50 transfer learning script
├── evaluate.py                              # Evaluation script for fine-tuned MobileNetV2
├── evaluate_efficientnet.py                 # Evaluation script for EfficientNetB0
├── evaluate_resnet50.py                     # Evaluation script for ResNet-50
├── check_model.py                           # Model initialization and verification check
├── training_history.csv                     # MobileNetV2 initial training history
├── finetuning_history.csv                   # MobileNetV2 fine-tuning history
├── efficientnetb0_history.csv               # EfficientNetB0 training history
├── resnet50_history.csv                     # ResNet-50 training history
├── efficientnet_classification_report.txt   # Verified empirical benchmark report (81.25% acc)
├── efficientnet_confusion_matrix.png        # Verified confusion matrix plot
├── resnet50_classification_report.txt       # Verified ResNet-50 evaluation report (79.75% acc)
├── resnet50_confusion_matrix.png            # Verified ResNet-50 confusion matrix plot
├── finetuned_mobilenetv2_report.txt         # Verified MobileNetV2 fine-tuning report (79.50% acc)
├── finetuned_mobilenetv2_confusion_matrix.png# Verified MobileNetV2 confusion matrix plot
├── pages/
│   ├── home.py                              # Landing page with dataset portfolio & workflow
│   ├── detection.py                         # Image upload, validation & multi-model inference
│   ├── results.py                           # Classification diagnosis, probabilities & Grad-CAM
│   └── model_comparison.py                 # 3-architecture benchmarks & Grad-CAM dashboard
├── components/
│   ├── navigation.py                        # AgriTech sidebar branding & session status
│   ├── cards.py                             # Metric cards & architectural summaries
│   └── charts.py                            # Interactive Plotly charts & confusion matrix heatmaps
├── utils/
│   ├── styling.py                           # AgriTech design system & CSS injection
│   ├── dataset.py                           # Dataset statistics & sample discovery
│   ├── model_loader.py                      # Cached TensorFlow / Keras model loader
│   ├── preprocessing.py                     # Image validation & unified tensor scaling
│   ├── prediction.py                        # Forward pass pipeline, latency & dynamic Grad-CAM
│   ├── metrics.py                           # Empirical results loader & evaluation parser
│   ├── gradcam.py                           # Dynamic tf.GradientTape Grad-CAM & overlays
│   └── report.py                            # Agronomic inspection text report generator
├── config/
│   └── settings.py                          # Central configurations, palette & class mapping
├── assets/
│   ├── styles.css                           # Custom AgriTech design system stylesheet
│   ├── logo.png                             # Brand mark
│   └── hero_leaf.jpg                        # High-resolution canopy photography
├── tests/
│   └── test_caulicare.py                    # Automated unit & integration test suite (8 tests)
├── data/
│   ├── README.md                            # Dataset layout documentation
│   └── samples/                             # Verified test specimens for live testing
│       ├── healthy_sample.jpg
│       ├── black_rot_sample.jpg
│       └── insect_hole_sample.jpg
├── models/
│   ├── README.md                            # Checkpoint specifications & instructions
│   ├── caulicare_finetuned.keras            # Production MobileNetV2 checkpoint
│   ├── efficientnetb0_best.keras            # EfficientNet-B0 checkpoint
│   └── resnet50_best.keras                  # ResNet-50 checkpoint
├── results/
│   ├── README.md                            # Guidelines for benchmark artifacts
│   ├── metrics/
│   │   └── model_metrics.json               # Structured empirical benchmark records
│   ├── confusion_matrices/                  # JSON confusion matrices
│   ├── training_history/                    # Epoch loss & accuracy histories
│   └── gradcam/                             # Verified Grad-CAM heatmaps & overlays
├── scripts/
│   └── build_models.py                      # Build, calibrate models & generate Grad-CAM
├── requirements.txt                         # Production dependencies
├── README.md                                # Project documentation
└── .gitignore                               # Git ignore patterns
```

---

## Setup & Launch Instructions (Windows PowerShell)

All commands are run from the project root directory:

### 1. Environment Setup
```powershell
# Optional: create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```powershell
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 3. Run Automated Tests
```powershell
python -m unittest tests/test_caulicare.py
```
Executes comprehensive tests covering image validation, model loading, empirical metrics parsing, multi-model inference, dynamic Grad-CAM generation, and agronomic reporting.

---

## Model Training & Evaluation Pipeline

### Step 1: Prepare Dataset Splits
Place the raw dataset inside:
`Dataset/Cauliflower Leaf Diseases A Computer Vision Dataset for Smart Agriculture/Processed Image/`
containing subfolders: `Healthy/`, `Black Rot/`, `Insect Hole/`.

Run data splitting (70% train, 15% validation, 15% test with stratification):
```powershell
python prepare_data.py
```
Generates `data_splits.csv`.

### Step 2: Train MobileNetV2 Baseline & Fine-Tuning
```powershell
# Initial training with frozen backbone
python train.py

# Fine-tune top 30 layers
python finetune.py

# Evaluate on held-out test split
python evaluate.py
```

### Step 3: Train & Evaluate ResNet-50
```powershell
# Train ResNet-50 with data augmentation and class weighting
python train_resnet50.py

# Evaluate on held-out test split
python evaluate_resnet50.py
```

### Step 4: Evaluate EfficientNet-B0
```powershell
python evaluate_efficientnet.py
```

### Step 5: Build / Calibrate Checkpoints & Generate Grad-CAM Artifacts
```powershell
python scripts/build_models.py
```

---

## Verified Empirical Benchmark Results

All three architectures were evaluated on the **exact same held-out test split of 400 images** (140 Healthy, 164 Black Rot, 96 Insect Hole) with identical class order:

| Architecture | Framework | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Model Size | Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B0** | TensorFlow / Keras | **81.25%** | **81.72%** | **81.42%** | **80.81%** | **81.39%** | 16.3 MB | 19.8 ms | Top Benchmark |
| **ResNet-50** | TensorFlow / Keras | **79.75%** | **80.33%** | **77.98%** | **78.60%** | **79.59%** | 90.7 MB | 28.4 ms | Evaluated Candidate |
| **MobileNetV2 (Fine-tuned)** | TensorFlow / Keras | **79.50%** | **78.80%** | **78.80%** | **78.80%** | **79.50%** | 9.2 MB | 14.2 ms | Production Edge Baseline |
| *MobileNetV2 (Baseline)* | TensorFlow / Keras | 77.00% | 76.70% | 77.32% | 76.07% | 77.24% | 9.2 MB | 14.2 ms | Reference Baseline |

### Per-Class Detailed Performance:
- **EfficientNet-B0:**
  - Healthy: Precision: 73.84%, Recall: 90.71%, F1: 81.41%
  - Black Rot: Precision: 96.06%, Recall: 74.39%, F1: 83.85%
  - Insect Hole: Precision: 75.25%, Recall: 79.17%, F1: 77.16%
- **ResNet-50:**
  - Healthy: Precision: 73.33%, Recall: 86.43%, F1: 79.34%
  - Black Rot: Precision: 85.00%, Recall: 82.93%, F1: 83.95%
  - Insect Hole: Precision: 82.67%, Recall: 64.58%, F1: 72.51%
- **MobileNetV2 (Fine-tuned):**
  - Healthy: Precision: 77.86%, Recall: 77.86%, F1: 77.86%
  - Black Rot: Precision: 83.54%, Recall: 83.54%, F1: 83.54%
  - Insect Hole: Precision: 75.00%, Recall: 75.00%, F1: 75.00%

### Deployment Selection Rationale:
- **MobileNetV2** is the research-oriented baseline cited in literature. With fine-tuning, it reaches 79.50% test accuracy with perfectly balanced class recall (75.0% on minority Insect Hole) and the smallest footprint (9.2 MB, 3.5M parameters), making it ideal for offline field scouting on smartphones.
- **EfficientNet-B0** achieved the highest overall empirical test accuracy (81.25%) and macro F1 (80.81%), making it the strongest candidate for high-throughput automated agricultural screening.
- **ResNet-50** is a high-capacity candidate (25.6M parameters, 90.7 MB) achieving 79.75% accuracy and strong Black Rot detection (82.93%), but with lower Insect Hole recall (64.58%) and 9.8× the storage footprint of MobileNetV2.

---

## Visual Explainability (Grad-CAM)

The application computes dynamic gradient-weighted class activation maps using **TensorFlow `GradientTape`**:
1. Locates the final convolutional layer (`out_relu` in MobileNetV2, `top_activation` in EfficientNet-B0, `conv5_block3_out` in ResNet-50).
2. Computes the gradient of the predicted class score with respect to the feature map activations.
3. Applies global average pooling and ReLU rectification to isolate positive attributions.
4. Normalizes and overlays the resulting heat map onto the original leaf photograph using the Jet colormap.

*Note on Saliency Interpretation:* Attention highlights represent statistical feature attribution within convolutional feature extractors; they indicate which leaf lamina sectors drove the classification decision, rather than biochemical proof of infection.

---

## Known Limitations & Agronomic Guidance
- **Field Illumination & Glare**: Strong direct sunlight or deep shadow can alter leaf chromaticity. For best results, capture images in diffuse natural daylight.
- **Nutritional Mimics**: Severe nitrogen or boron deficiencies can manifest marginal chlorosis visually similar to early Black Rot. Confirm large-scale infections with local agricultural extension plant pathology diagnostics.
- **AI-Assisted Screening**: CauliCare is designed as an agronomic decision-support and screening aid; critical field management interventions should always be corroborated with physical scouting.

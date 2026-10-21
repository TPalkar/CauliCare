# CauliCare AI — Intelligent Cauliflower Leaf Disease Detection

[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.22-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-TPalkar%2FCauliCare-181717?logo=github&logoColor=white)](https://github.com/TPalkar/CauliCare.git)

**CauliCare AI** is an end-to-end precision agriculture platform designed for automated pathological screening of cauliflower (*Brassica oleracea*) crops. The application integrates deep convolutional neural networks (**TensorFlow / Keras MobileNetV2** production pipeline, alongside **EfficientNet-B0** benchmark evaluation), empirical performance dashboards, visual saliency maps (**Grad-CAM**), and science-backed agronomic guidance.

---

## 📑 Scientific & Dataset References
- **Research Paper:** [PMC Article PMC12175232 — Deep Learning for Cauliflower Disease Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175232/)
- **Dataset Archive:** [Mendeley Data — Cauliflower Leaf Dataset (v1)](https://data.mendeley.com/datasets/x995snz7p3/1)
- **GitHub Repository:** [https://github.com/TPalkar/CauliCare.git](https://github.com/TPalkar/CauliCare.git) (`main` branch)

---

## 🎯 Unified Diagnostic Scope & Class Mapping

The model classifies leaves into three distinct phytopathological conditions with an explicit, unified index mapping across data preparation, training, fine-tuning, evaluation, and user interface:

| Index | Class Name | Biological Description | Severity Protocol |
| :---: | :--- | :--- | :--- |
| **0** | **Healthy** | Intact foliage, uniform emerald pigmentation, no visible chlorotic lesions or perforations. | Routine Monitoring |
| **1** | **Black Rot** | Critical bacterial infection (*Xanthomonas campestris* pv. *campestris*). V-shaped chlorotic lesions with darkened veins. | Urgent Containment |
| **2** | **Insect Hole** | Foliar chewing damage, shot-hole apertures, and windowpaning caused by chewing larvae (e.g. *Plutella xylostella*). | Action Advised (48h) |

---

## 🏛️ Repository Structure

```text
CauliCare/
├── app.py                                   # Streamlit application entrypoint (st.navigation)
├── prepare_data.py                          # Stratified dataset split generator (train/val/test)
├── train.py                                 # MobileNetV2 transfer learning with class weighting
├── finetune.py                              # MobileNetV2 fine-tuning (top 30 layers unfrozen)
├── evaluate.py                              # Evaluation script for fine-tuned MobileNetV2
├── evaluate_efficientnet.py                 # Evaluation script for EfficientNetB0
├── check_model.py                           # Model initialization and verification check
├── efficientnet_classification_report.txt   # Verified empirical benchmark report (81.25% acc)
├── efficientnet_confusion_matrix.png        # Verified confusion matrix plot
├── pages/
│   ├── home.py                              # Landing page with dataset portfolio & charts
│   ├── detection.py                         # Image upload, validation & inference trigger
│   ├── results.py                           # Classification diagnosis, probabilities & report
│   └── model_comparison.py                 # 3-architecture benchmarks & Grad-CAM dashboard
├── components/
│   ├── navigation.py                        # AgriTech sidebar branding & session status
│   ├── cards.py                             # Metric cards & architectural summaries
│   ├── charts.py                            # Interactive Plotly charts & confusion matrix heatmaps
│   └── image_gallery.py                     # Pathology specimen visual gallery
├── utils/
│   ├── styling.py                           # AgriTech design system & CSS injection
│   ├── dataset.py                           # Dataset statistics & sample discovery
│   ├── model_loader.py                      # Cached TensorFlow / Keras model loader
│   ├── preprocessing.py                     # Image validation & MobileNetV2 tensor scaling
│   ├── prediction.py                        # Forward pass pipeline & calibrated probabilities
│   ├── metrics.py                           # Empirical results loader & evaluation parser
│   ├── gradcam.py                           # TensorFlow Keras Grad-CAM hooks & saliency
│   └── report.py                            # Agronomic inspection text report generator
├── config/
│   └── settings.py                          # Central configurations, palette & class mapping
├── assets/
│   ├── styles.css                           # Custom AgriTech design system stylesheet
│   ├── logo.png                             # Brand mark
│   └── hero_leaf.jpg                        # High-resolution canopy photography
├── tests/
│   └── test_caulicare.py                    # Automated unit & integration test suite
├── data/
│   ├── README.md                            # Dataset layout documentation
│   └── samples/                             # Verified test specimens for live testing
│       ├── healthy_sample.jpg
│       ├── black_rot_sample.jpg
│       └── insect_hole_sample.jpg
├── models/
│   └── README.md                            # Checkpoint specifications & instructions
├── results/
│   ├── README.md                            # Guidelines for benchmark artifacts
│   ├── metrics/
│   │   └── model_metrics.json               # Structured empirical benchmark records
│   ├── confusion_matrices/                  # JSON confusion matrices
│   ├── training_history/                    # Epoch loss & accuracy histories
│   └── gradcam/                             # Saved Grad-CAM saliency heatmaps
├── requirements.txt                         # Production dependencies
├── README.md                                # Project documentation
└── .gitignore                               # Git ignore patterns
```

---

## 🚀 Setup & Launch Instructions (Windows PowerShell)

All commands are run from the project root directory (`C:\Users\Arya\Documents\CauliCare-repo`):

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
Executes tests covering preprocessing, class mapping, missing model fallback, empirical metrics parsing, Grad-CAM safety, and report generation.

---

## 🔬 Model Training & Evaluation Pipeline

### Step 1: Prepare Dataset Splits
Place the raw dataset inside:
`Dataset/Cauliflower Leaf Diseases A Computer Vision Dataset for Smart Agriculture/Processed Image/`
containing subfolders: `Healthy/`, `Black Rot/`, `Insect Hole/`.

Run data splitting (70% train, 15% validation, 15% test with stratification):
```powershell
python prepare_data.py
```
Generates `data_splits.csv`.

### Step 2: Train MobileNetV2 Baseline
```powershell
python train.py
```
Trains MobileNetV2 with ImageNet weights, class weighting for imbalance mitigation, and early stopping. Saves:
- `models/caulicare_best.keras`
- `models/caulicare_final.keras`
- `training_history.csv`

### Step 3: Fine-Tune MobileNetV2
```powershell
python finetune.py
```
Unfreezes the top 30 layers with learning rate $1 \times 10^{-5}$. Saves:
- `models/caulicare_finetuned.keras` (Active production model)
- `finetuning_history.csv`

### Step 4: Evaluate Models
```powershell
# Evaluate MobileNetV2 on the test set:
python evaluate.py

# Evaluate EfficientNetB0 on the test set:
python evaluate_efficientnet.py
```
Saves `confusion_matrix.png`, prints test loss, test accuracy, and classification reports.

---

## 📊 Verified Empirical Benchmark Results

The repository includes verified test set benchmarks on held-out cauliflower test samples:

| Model Architecture | Framework | Test Accuracy | Precision (Wtd) | Recall (Wtd) | F1-Score (Wtd) | Test Support | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B0** | TensorFlow / Keras | **81.25%** | **83.29%** | **81.25%** | **81.39%** | 400 images | Verified Benchmark |
| **MobileNetV2** | TensorFlow / Keras | *Pending Training* | *Pending* | *Pending* | *Pending* | — | Production Pipeline |
| **ResNet-50** | TensorFlow / Keras | *Pending Training* | *Pending* | *Pending* | *Pending* | — | Candidate Architecture |

*Detailed per-class metrics for EfficientNet-B0:*
- **Healthy**: Precision: 73.84%, Recall: 90.71%, F1-Score: 81.41% (140 samples)
- **Black Rot**: Precision: 96.06%, Recall: 74.39%, F1-Score: 83.85% (164 samples)
- **Insect Hole**: Precision: 75.25%, Recall: 79.17%, F1-Score: 77.16% (96 samples)

---

## 🔍 Visual Explainability (Grad-CAM)

The application computes gradient-weighted class activation maps using **TensorFlow `GradientTape`**:
1. Locates the final convolutional layer (e.g. `Conv_1` in MobileNetV2).
2. Computes the gradient of the predicted class score with respect to the feature map activations.
3. Applies global average pooling and ReLU rectification.
4. Overlays the resulting heat map onto the original leaf photograph using the Jet colormap.

*Note on Saliency Interpretation:* Attention highlights represent statistical feature attribution within the convolutional kernels; they do not constitute clinical biochemical proof of bacterial infection etiology.

---

## ⚠️ Known Limitations & Agronomic Guidance
- **Field Illumination & Glare**: Strong direct sunlight or deep shadow can alter leaf chromaticity. For best results, capture images in diffuse natural daylight.
- **Nutritional Mimics**: Severe nitrogen or boron deficiencies can manifest marginal chlorosis visually similar to early Black Rot. Confirm large-scale infections with local agricultural extension plant pathology diagnostics.

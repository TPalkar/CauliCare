"""
CauliCare - Central Configuration and Settings
Unified configuration for TensorFlow / Keras pipeline, class mapping,
dataset statistics, model architectures, and AgriTech visual design tokens.
"""

from pathlib import Path
from typing import Dict, Any, List

# ==============================================================================
# BASE DIRECTORIES (Works from repository root)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
HISTORY_DIR = RESULTS_DIR / "training_history"
GRADCAM_DIR = RESULTS_DIR / "gradcam"
SAMPLES_DIR = DATA_DIR / "samples"

# ==============================================================================
# EXTERNAL REFERENCES
# ==============================================================================
RESEARCH_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12175232/"
DATASET_SOURCE_URL = "https://data.mendeley.com/datasets/x995snz7p3/1"
GITHUB_REPO_URL = "https://github.com/TPalkar/CauliCare.git"

# ==============================================================================
# UNIFIED CLASS MAPPING (Explicitly matching train.py & evaluate.py)
# ==============================================================================
CLASS_NAMES: List[str] = ["Healthy", "Black Rot", "Insect Hole"]
NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}
INDEX_TO_CLASS: Dict[int, str] = {
    idx: name for idx, name in enumerate(CLASS_NAMES)
}

# ==============================================================================
# DATASET CONFIGURATION
# Verified class counts reported in literature & repository
# ==============================================================================
DATASET_COUNTS: Dict[str, int] = {
    "Healthy": 934,
    "Black Rot": 1088,
    "Insect Hole": 639,
}

TOTAL_IMAGES = sum(DATASET_COUNTS.values())
HEALTHY_COUNT = DATASET_COUNTS["Healthy"]
DISEASED_COUNT = DATASET_COUNTS["Black Rot"] + DATASET_COUNTS["Insect Hole"]

CLASS_PERCENTAGES = {
    cls: (count / TOTAL_IMAGES) * 100 for cls, count in DATASET_COUNTS.items()
}
HEALTHY_PERCENTAGE = (HEALTHY_COUNT / TOTAL_IMAGES) * 100
DISEASED_PERCENTAGE = (DISEASED_COUNT / TOTAL_IMAGES) * 100

# ==============================================================================
# COLOR PALETTE & DESIGN SYSTEM TOKENS
# ==============================================================================
THEME = {
    "primary": "#173F35",        # Deep Forest Green
    "secondary": "#2F6B4F",      # Foliage Green
    "light_leaf": "#D8E8C8",     # Fresh Leaf Green
    "cream_bg": "#F5F4E9",       # Warm Cream Canvas
    "card_bg": "#FFFFFF",        # Crisp Card Background
    "text_main": "#1E2923",      # Near-black organic text
    "text_muted": "#68736B",     # Secondary Slate Green
    "border": "#E2E7DF",         # Subtle card border
    "border_focus": "#2F6B4F",
    "error": "#B91C1C",          # Muted earthy red
    "warning": "#D97706",        # Muted harvest amber
    "success": "#15803D",        # Vibrant healthy green
}

CHART_COLORS = {
    "Healthy": "#2F6B4F",
    "Black Rot": "#831843",
    "Insect Hole": "#D97706",
    "Diseased": "#9F1239",
    "MobileNetV2": "#2F6B4F",
    "EfficientNet": "#173F35",
    "ResNet": "#4D7C0F",
}

# ==============================================================================
# MODEL ARCHITECTURES & SETTINGS (Primary: TensorFlow/Keras)
# ==============================================================================
ML_FRAMEWORK = "tensorflow"
SUPPORTED_MODELS = ["MobileNetV2", "EfficientNet", "ResNet"]
DEFAULT_PRODUCTION_MODEL = "MobileNetV2"

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MobileNetV2": {
        "name": "MobileNetV2",
        "description": "Production edge-optimized inverted residual network from train.py / finetune.py.",
        "framework": "tensorflow",
        "input_size": (224, 224),
        "target_conv_layer": "Conv_1",
        "checkpoints": [
            MODELS_DIR / "caulicare_finetuned.keras",
            MODELS_DIR / "caulicare_best.keras",
            MODELS_DIR / "caulicare_final.keras",
        ],
        "default_checkpoint": MODELS_DIR / "caulicare_finetuned.keras",
    },
    "EfficientNet": {
        "name": "EfficientNet-B0",
        "description": "Compound scaled convolutional network evaluated on 400 test images.",
        "framework": "tensorflow",
        "input_size": (224, 224),
        "target_conv_layer": "top_conv",
        "checkpoints": [
            MODELS_DIR / "efficientnetb0_best.keras",
        ],
        "default_checkpoint": MODELS_DIR / "efficientnetb0_best.keras",
    },
    "ResNet": {
        "name": "ResNet-50",
        "description": "Deep residual learning framework with skip connections.",
        "framework": "tensorflow",
        "input_size": (224, 224),
        "target_conv_layer": "conv5_block3_out",
        "checkpoints": [
            MODELS_DIR / "resnet50_cauliflower.keras",
        ],
        "default_checkpoint": MODELS_DIR / "resnet50_cauliflower.keras",
    },
}

# ==============================================================================
# DISEASE CLINICAL METADATA & AGRONOMIC PROTOCOLS
# ==============================================================================
DISEASE_INFO: Dict[str, Dict[str, Any]] = {
    "Healthy": {
        "title": "Healthy Foliage",
        "status_badge": "Normal / Healthy",
        "badge_color": "success",
        "severity": "None",
        "overview": (
            "The leaf exhibits uniform green pigmentation, intact cellular margins, and "
            "no visible chlorotic lesions or pest perforations."
        ),
        "symptoms": [
            "Uniform emerald to forest-green coloration across the lamina",
            "Smooth leaf surface without necrotic spots or yellow halos",
            "Intact midrib and lateral venation",
            "No evidence of larval feeding apertures or bacterial ooze"
        ],
        "recommendations": [
            "Continue standard agronomic practices and balanced N-P-K nutrient management.",
            "Maintain scheduled drip or furrow irrigation without waterlogging root zones.",
            "Perform weekly scouting for early signs of diamondback moth or aphids.",
            "Ensure adequate field aeration and crop spacing (45–60 cm between rows)."
        ],
        "urgency": "Routine Monitoring"
    },
    "Black Rot": {
        "title": "Black Rot (Xanthomonas campestris)",
        "status_badge": "Critical Bacterial Infection",
        "badge_color": "error",
        "severity": "High (Systemic Pathogen)",
        "overview": (
            "Black rot, caused by the bacterium Xanthomonas campestris pv. campestris (Xcc), "
            "is one of the most destructive crucifer diseases worldwide. It enters through leaf "
            "hydathodes or wounds, colonizing vascular bundles and causing V-shaped chlorotic lesions."
        ),
        "symptoms": [
            "Characteristic V-shaped chlorotic (yellow) to necrotic lesions at leaf margins",
            "Base of the 'V' pointing toward the leaf midrib or primary veins",
            "Darkened, blackened vascular veins visible upon leaf inspection",
            "Premature defoliation, stunted curds, and systemic wilting"
        ],
        "recommendations": [
            "Immediately isolate and carefully rogue out severely infected plants to prevent field spread.",
            "Avoid overhead irrigation, which splashes bacterial cells between adjacent leaves.",
            "Disinfect pruning tools and field machinery with recommended sanitizing solutions.",
            "Implement a minimum 2-to-3-year crop rotation with non-brassica crops.",
            "Seek guidance from regional agricultural extension services for certified disease management protocols."
        ],
        "urgency": "Urgent Containment"
    },
    "Insect Hole": {
        "title": "Insect Foliar Damage",
        "status_badge": "Pest Damage Detected",
        "badge_color": "warning",
        "severity": "Moderate (Foliar)",
        "overview": (
            "Visible leaf perforations and ragged margins typically caused by chewers "
            "such as Diamondback Moth (Plutella xylostella), Cabbage Looper, or flea beetles. "
            "Foliar loss diminishes photosynthetic capacity during head formation."
        ),
        "symptoms": [
            "Irregular shot-hole perforations across interveinal leaf tissue",
            "Windowpane feeding marks where epidermis remains translucent",
            "Presence of dark frass (insect fecal matter) near feeding sites",
            "Skeletonized or scalloped leaf borders"
        ],
        "recommendations": [
            "Perform direct inspection under the abaxial (underside) leaf surfaces for larvae or eggs.",
            "Install yellow sticky traps and pheromone traps to quantify adult pest pressure.",
            "Encourage natural predators including parasitic wasps and hoverfly larvae.",
            "Consult local agricultural extension specialists for approved biological or targeted IPM interventions."
        ],
        "urgency": "Action Advised (Within 48h)"
    }
}

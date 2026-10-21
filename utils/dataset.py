"""
Dataset statistics, metadata access, and sample image discovery for CauliCare.
"""

from pathlib import Path
from typing import Dict, List, Optional
from config.settings import (
    DATASET_COUNTS,
    TOTAL_IMAGES,
    HEALTHY_COUNT,
    DISEASED_COUNT,
    CLASS_PERCENTAGES,
    HEALTHY_PERCENTAGE,
    DISEASED_PERCENTAGE,
    SAMPLES_DIR,
    DATA_DIR,
    RESEARCH_PAPER_URL,
    DATASET_SOURCE_URL,
)


def get_dataset_stats() -> Dict[str, any]:
    """Returns verified dataset counts and derived proportions."""
    return {
        "total": TOTAL_IMAGES,
        "classes_count": len(DATASET_COUNTS),
        "healthy": HEALTHY_COUNT,
        "diseased": DISEASED_COUNT,
        "counts": DATASET_COUNTS,
        "percentages": CLASS_PERCENTAGES,
        "healthy_percentage": HEALTHY_PERCENTAGE,
        "diseased_percentage": DISEASED_PERCENTAGE,
    }


def discover_sample_images() -> Dict[str, Optional[Path]]:
    """
    Discovers representative sample images for Healthy, Insect Hole, and Black Rot.
    Checks data/samples first, then dataset directory structures.
    """
    samples = {"Healthy": None, "Insect Hole": None, "Black Rot": None}

    # Standard candidate naming in samples directory
    candidates = {
        "Healthy": [
            SAMPLES_DIR / "healthy_sample.jpg",
            SAMPLES_DIR / "healthy.jpg",
            DATA_DIR / "Healthy" / "sample.jpg",
        ],
        "Insect Hole": [
            SAMPLES_DIR / "insect_hole_sample.jpg",
            SAMPLES_DIR / "insect_hole.jpg",
            DATA_DIR / "Insect Hole" / "sample.jpg",
        ],
        "Black Rot": [
            SAMPLES_DIR / "black_rot_sample.jpg",
            SAMPLES_DIR / "black_rot.jpg",
            DATA_DIR / "Black Rot" / "sample.jpg",
        ],
    }

    for label, paths in candidates.items():
        for p in paths:
            if p.exists() and p.is_file():
                samples[label] = p
                break

    # If not found yet, inspect any image files inside category folders
    for label in samples:
        if samples[label] is None:
            folder = DATA_DIR / label
            if folder.exists() and folder.is_dir():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    files = list(folder.glob(ext))
                    if files:
                        samples[label] = files[0]
                        break

    return samples


def get_external_links() -> Dict[str, str]:
    """Returns references to the research paper and original dataset."""
    return {
        "paper_url": RESEARCH_PAPER_URL,
        "dataset_url": DATASET_SOURCE_URL,
    }

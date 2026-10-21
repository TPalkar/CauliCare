
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_DIR
    / "Dataset"
    / "Cauliflower Leaf Diseases A Computer Vision Dataset for Smart Agriculture"
    / "Processed Image"
)

OUTPUT_FILE = PROJECT_DIR / "data_splits.csv"

CLASSES = ["Healthy", "Black Rot", "Insect Hole"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42


def main():
    records = []

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}"
            )

        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        if not images:
            raise ValueError(f"No images found in {class_dir}")

        for image_path in images:
            records.append({
                "filepath": str(image_path.relative_to(PROJECT_DIR)),
                "label": class_name,
            })

        print(f"{class_name}: {len(images)} images")

    df = pd.DataFrame(records)

    # First reserve 30% for validation and testing.
    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    # Divide the reserved 30% equally into validation and testing.
    val_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    result = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSplit totals:")
    print(result["split"].value_counts().to_string())

    print("\nClass distribution by split:")
    print(pd.crosstab(result["label"], result["split"]))

    print(f"\nSaved manifest to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
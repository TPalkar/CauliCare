
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data_splits.csv"
MODEL_PATH = PROJECT_DIR / "models" / "caulicare_best.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = ["Healthy", "Black Rot", "Insect Hole"]
CLASS_TO_INDEX = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

# Separate output files for the original MobileNetV2 model
OUTPUT_PNG = PROJECT_DIR / "mobilenetv2_confusion_matrix.png"
OUTPUT_REPORT = PROJECT_DIR / "mobilenetv2_classification_report.txt"

# --------------------------------------------------
# 2. Validate input files
# --------------------------------------------------
if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset manifest not found: {CSV_PATH}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Original MobileNetV2 model not found: {MODEL_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {"filepath", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required_columns}"
    )

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found in data_splits.csv.")

if not test_df["label"].isin(CLASS_NAMES).all():
    raise ValueError("Unexpected class label found in the dataset.")

for relative_path in test_df["filepath"]:
    if not (PROJECT_DIR / relative_path).is_file():
        raise FileNotFoundError(
            f"Image not found: {relative_path}"
        )

print("Model:", MODEL_PATH)
print("Test images:", len(test_df))

def load_image(relative_path, label):
    full_path = tf.strings.join(
        [str(PROJECT_DIR) + "/", relative_path]
    )

    image_bytes = tf.io.read_file(full_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # Correct preprocessing for MobileNetV2
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


paths = test_df["filepath"].to_numpy(dtype=str)

y_true = test_df["label"].map(CLASS_TO_INDEX).to_numpy(
    dtype=np.int32
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (paths, y_true)
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

print("\nLoading original MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating test set...")
loss, accuracy = model.evaluate(test_ds, verbose=1)

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\n" + "=" * 50)
print("ORIGINAL MOBILENETV2 EVALUATION")
print("=" * 50)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\nClassification report:")
print(report)

with open(OUTPUT_REPORT, "w", encoding="utf-8") as file:
    file.write("Original MobileNetV2 Evaluation\n\n")
    file.write(f"Test loss: {loss:.4f}\n")
    file.write(
        f"Test accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)\n\n"
    )
    file.write(report)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
)

fig, ax = plt.subplots(figsize=(8, 6))

matrix_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

matrix_display.plot(
    ax=ax,
    values_format="d",
    xticks_rotation=25,
    colorbar=False,
)

ax.set_title("CauliCare - Original MobileNetV2 Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

fig.tight_layout()
fig.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

print("\nConfusion matrix:")
print(cm)

print("\nFiles saved:")
print("Classification report:", OUTPUT_REPORT)
print("Confusion matrix:", OUTPUT_PNG)
print("Report exists:", OUTPUT_REPORT.is_file())
print("PNG exists:", OUTPUT_PNG.is_file())
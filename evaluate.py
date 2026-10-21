
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

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data_splits.csv"
MODEL_PATH = PROJECT_DIR / "models" / "caulicare_finetuned.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["Healthy", "Black Rot", "Insect Hole"]

if not CSV_PATH.exists():
    raise FileNotFoundError("data_splits.csv not found.")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Best model not found. Run train.py successfully first."
    )

df = pd.read_csv(CSV_PATH)
test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found in data_splits.csv.")

class_to_index = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

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

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


paths = test_df["filepath"].to_numpy(dtype=str)
y_true = test_df["label"].map(class_to_index).to_numpy(
    dtype=np.int32
)

test_ds = tf.data.Dataset.from_tensor_slices((paths, y_true))
test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = model.evaluate(test_ds, verbose=1)

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)

display.plot(values_format="d", xticks_rotation=25)
plt.title("CauliCare - Test Set Confusion Matrix")
plt.tight_layout()

output_path = PROJECT_DIR / "confusion_matrix.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"\nConfusion matrix saved to: {output_path}")
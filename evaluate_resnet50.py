
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ---------------- CONFIGURATION ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CSV_PATH = "data_splits.csv"
MODEL_PATH = "models/resnet50_best.keras"

CM_PATH = "resnet50_confusion_matrix.png"
REPORT_PATH = "resnet50_classification_report.txt"

CLASS_NAMES = ["Healthy", "Black Rot", "Insect Hole"]


def load_and_prepare_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


# ---------------- LOAD TEST DATA ----------------
df = pd.read_csv(CSV_PATH)

required_columns = {"filepath", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"{CSV_PATH} must contain columns: {required_columns}"
    )

project_root = os.path.dirname(os.path.abspath(CSV_PATH))
df["filepath"] = df["filepath"].apply(
    lambda p: p if os.path.isabs(p)
    else os.path.join(project_root, p)
)

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test rows found in data_splits.csv.")

label_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

unknown_labels = set(test_df["label"].unique()) - set(CLASS_NAMES)
if unknown_labels:
    raise ValueError(f"Unexpected labels in test set: {unknown_labels}")

test_df["label_id"] = (
    test_df["label"].map(label_to_id).astype("int32")
)

missing = test_df.loc[
    ~test_df["filepath"].apply(os.path.isfile), "filepath"
]
if not missing.empty:
    raise FileNotFoundError(
        f"{len(missing)} test images are missing. "
        f"Example: {missing.iloc[0]}"
    )

test_ds = tf.data.Dataset.from_tensor_slices((
    test_df["filepath"].values,
    test_df["label_id"].values.astype("int32")
))
test_ds = test_ds.map(
    load_and_prepare_image,
    num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---------------- LOAD MODEL ----------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run train_resnet50.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate using the model's built-in preprocessing.
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

# ---------------- PREDICTIONS ----------------
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)
y_true = test_df["label_id"].values

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES))
)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n========== RESNET50 TEST RESULTS ==========")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Prediction accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:\n")
print(report)
print("Confusion matrix:\n", cm)

# ---------------- SAVE REPORT ----------------
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("ResNet50 - CauliCare Test Evaluation\n")
    file.write("=" * 42 + "\n")
    file.write(f"Test images: {len(test_df)}\n")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
    file.write("Class order: " + ", ".join(CLASS_NAMES) + "\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(np.array2string(cm))

# ---------------- SAVE CONFUSION MATRIX ----------------
fig, ax = plt.subplots(figsize=(8, 6))
image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted label",
    ylabel="True label",
    title="ResNet50 - Confusion Matrix"
)

plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

threshold = cm.max() / 2 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

fig.tight_layout()
fig.savefig(CM_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved files:")
print("Confusion matrix:", CM_PATH)
print("Classification report:", REPORT_PATH)
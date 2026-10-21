
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# -----------------------------
# Configuration
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data_splits.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

CLASS_NAMES = ["Healthy", "Black Rot", "Insect Hole"]
NUM_CLASSES = len(CLASS_NAMES)

tf.keras.utils.set_random_seed(SEED)

# -----------------------------
# Load and validate manifest
# -----------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CSV_PATH}. Run prepare_data.py first."
    )

df = pd.read_csv(CSV_PATH)

required_columns = {"filepath", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV must contain: {required_columns}")

for relative_path in df["filepath"]:
    if not (PROJECT_DIR / relative_path).is_file():
        raise FileNotFoundError(
            f"Image not found: {relative_path}"
        )

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation split is empty.")

class_to_index = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

for label in df["label"].unique():
    if label not in class_to_index:
        raise ValueError(f"Unexpected class: {label}")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Classes:", CLASS_NAMES)

# -----------------------------
# Image loading pipeline
# -----------------------------
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

    # Required preprocessing for MobileNetV2.
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


def make_dataset(frame, training=False):
    paths = frame["filepath"].to_numpy(dtype=str)
    labels = frame["label"].map(class_to_index).to_numpy(
        dtype=np.int32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)

# -----------------------------
# Handle class imbalance
# -----------------------------
y_train = train_df["label"].map(class_to_index).to_numpy()

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train,
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(weights)
}

print("Class weights:", class_weights)

# -----------------------------
# Build MobileNetV2 model
# -----------------------------
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation",
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
best_model_path = MODEL_DIR / "caulicare_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save final training model and history.
model.save(MODEL_DIR / "caulicare_final.keras")

pd.DataFrame(history.history).to_csv(
    PROJECT_DIR / "training_history.csv",
    index=False,
)

print("\nTraining complete.")
print("Best model:", best_model_path)
print("Final model:", MODEL_DIR / "caulicare_final.keras")
print("History:", PROJECT_DIR / "training_history.csv")
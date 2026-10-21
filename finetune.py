
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "data_splits.csv"
INPUT_MODEL = PROJECT_DIR / "models" / "caulicare_best.keras"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_MODEL = MODEL_DIR / "caulicare_finetuned.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

CLASS_NAMES = ["Healthy", "Black Rot", "Insect Hole"]
NUM_CLASSES = len(CLASS_NAMES)

tf.keras.utils.set_random_seed(SEED)

# 1. Check required files
if not CSV_PATH.exists():
    raise FileNotFoundError("data_splits.csv not found.")

if not INPUT_MODEL.exists():
    raise FileNotFoundError(
        f"Baseline model not found: {INPUT_MODEL}"
    )

df = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

class_to_index = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

# 2. Load images with the same preprocessing as training
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


def make_dataset(frame, training=False):
    paths = frame["filepath"].to_numpy(dtype=str)
    labels = frame["label"].map(class_to_index).to_numpy(
        dtype=np.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(frame),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)

# 3. Calculate class weights, as in initial training
y_train = train_df["label"].map(class_to_index).to_numpy()

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train,
)
class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(weights)
}

# 4. Load the best baseline model
model = tf.keras.models.load_model(
    INPUT_MODEL,
    compile=False,
)

# Find the nested MobileNetV2 feature extractor.
base_model = next(
    (
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "mobilenetv2" in layer.name.lower()
    ),
    None,
)

if base_model is None:
    raise ValueError(
        "Could not find the MobileNetV2 base model. "
        "Inspect model.summary() before proceeding."
    )

# 5. Unfreeze only the final 30 layers.
# Keep BatchNormalization layers frozen for stability.
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

print("Trainable MobileNetV2 layers:")
for layer in base_model.layers[-30:]:
    if layer.trainable:
        print(" ", layer.name)

# 6. Recompile with a low learning rate.
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 7. Save the best model based on validation loss.
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_MODEL),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

# 8. Fine-tune
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save history for later comparison.
pd.DataFrame(history.history).to_csv(
    PROJECT_DIR / "finetuning_history.csv",
    index=False,
)

print("\nFine-tuning finished.")
print("Best fine-tuned model:", OUTPUT_MODEL)
print("History:", PROJECT_DIR / "finetuning_history.csv")
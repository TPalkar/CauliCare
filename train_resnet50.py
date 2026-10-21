
import os
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ---------------- CONFIGURATION ----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

CSV_PATH = "data_splits.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_best.keras")
FINAL_PATH = os.path.join(MODEL_DIR, "resnet50_final.keras")
HISTORY_PATH = "resnet50_history.csv"

CLASS_NAMES = ["Healthy", "Black Rot", "Insect Hole"]
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_and_prepare_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(frame, training=False):
    paths = frame["filepath"].values
    labels = frame["label_id"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(frame),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_and_prepare_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------- LOAD DATA SPLITS ----------------
df = pd.read_csv(CSV_PATH)

required_columns = {"filepath", "label", "split"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"{CSV_PATH} must contain columns: {required_columns}"
    )

# Resolve relative paths from the project root.
project_root = os.path.dirname(os.path.abspath(CSV_PATH))
df["filepath"] = df["filepath"].apply(
    lambda p: p if os.path.isabs(p)
    else os.path.join(project_root, p)
)

missing = df.loc[~df["filepath"].apply(os.path.isfile), "filepath"]
if not missing.empty:
    raise FileNotFoundError(
        f"{len(missing)} image paths do not exist. "
        f"Example: {missing.iloc[0]}"
    )

unknown_labels = set(df["label"].unique()) - set(CLASS_NAMES)
if unknown_labels:
    raise ValueError(f"Unexpected labels in CSV: {unknown_labels}")

label_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
df["label_id"] = df["label"].map(label_to_id).astype("int32")

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

if train_df.empty or val_df.empty:
    raise ValueError(
        "No train or validation rows found. Check data_splits.csv."
    )

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Class order:", CLASS_NAMES)
print(train_df["label"].value_counts())

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)

# Calculate class weights from the training split only.
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=train_df["label_id"].values
)
class_weights = {
    i: float(weight) for i, weight in enumerate(weights)
}
print("Class weights:", class_weights)

# ---------------- BUILD RESNET50 ----------------
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES), activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(HISTORY_PATH)
]

# ---------------- TRAIN ----------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(FINAL_PATH)

print("\nTraining complete.")
print("Best checkpoint:", MODEL_PATH)
print("Final model:", FINAL_PATH)
print("Training history:", HISTORY_PATH)

import tensorflow as tf

print("TensorFlow:", tf.__version__)

model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

print("MobileNetV2 loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
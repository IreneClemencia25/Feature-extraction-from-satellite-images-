# feature_extraction.py

import tensorflow as tf
import numpy as np
import os

# -------------------------
# 1. Dataset configuration
# -------------------------
data_dir = r"D:\Feature_extraction_from_satellite_images final\data\EuroSAT_RGB"
batch_size = 16  # smaller batch size to reduce memory usage
img_size = (64, 64)

# -------------------------
# 2. Load dataset
# -------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    shuffle=True,
    batch_size=16,   # smaller batch size
    image_size=img_size
)


class_names = dataset.class_names
print("Classes:", class_names)

# -------------------------
# 3. Preprocessing
# -------------------------
# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)
dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

# -------------------------
# 4. Feature extraction using ResNet50
# -------------------------
# Load ResNet50 pretrained on ImageNet, exclude top classification layer
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)

# Freeze the model to prevent training
base_model.trainable = False

# Create feature extraction model
feature_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

# -------------------------
# 5. Extract features
# -------------------------
features_list = []
labels_list = []

for images, labels in dataset:
    features = feature_model(images)
    features_list.append(features.numpy())
    labels_list.append(labels.numpy())

# Convert lists to arrays
features_array = np.concatenate(features_list, axis=0)
labels_array = np.concatenate(labels_list, axis=0)

print("Feature shape:", features_array.shape)
print("Labels shape:", labels_array.shape)

# -------------------------
# 6. Save features and labels
# -------------------------
np.save("features.npy", features_array)
np.save("labels.npy", labels_array)
print("Features and labels saved successfully.")

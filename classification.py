# ============================================================
# EuroSAT RGB – MobileNetV2 Feature Extraction + SVM
# FAST & CPU-FRIENDLY VERSION
# ============================================================

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "EuroSAT_RGB")

IMG_SIZE = (192, 192)
BATCH_SIZE = 32
RANDOM_STATE = 42

USE_SAVED_FEATURES = True   # <<< VERY IMPORTANT

FEATURES_PATH = os.path.join(BASE_DIR, "features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")

# -------------------------------
# LOAD DATASET
# -------------------------------
print("[INFO] Dataset path:", DATASET_DIR)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    label_mode="int"
)

class_names = dataset.class_names
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# -------------------------------
# LOAD FEATURE EXTRACTOR (FAST)
# -------------------------------
print("[INFO] Loading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(192, 192, 3)
)
base_model.trainable = False

feature_extractor = tf.keras.Sequential([
    tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input
    ),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

# -------------------------------
# FEATURE EXTRACTION (ONCE)
# -------------------------------
if USE_SAVED_FEATURES and os.path.exists(FEATURES_PATH):
    print("[INFO] Loading saved features...")
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)

else:
    print("[INFO] Extracting features (this may take time ONCE)...")

    features_list = []
    labels_list = []

    for images, lbls in dataset:
        batch_features = feature_extractor.predict(images, verbose=0)
        features_list.append(batch_features)
        labels_list.append(lbls.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels)

    print("[INFO] Features saved for future runs")

print("[INFO] Feature shape:", features.shape)
print("[INFO] Label shape:", labels.shape)

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels
)

# -------------------------------
# TRAIN SVM
# -------------------------------
print("[INFO] Training SVM...")

svm = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

svm.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
print("[INFO] Evaluating model...")

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nOverall Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nPer-Class Accuracy:")
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(per_class_accuracy):
    print(f"{class_names[i]:<25}: {acc:.4f}")

# -------------------------------
# CONFUSION MATRIX PLOT
# -------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – EuroSAT RGB (MobileNetV2 + SVM)")
plt.tight_layout()
plt.show()

print("\n[INFO] DONE.")


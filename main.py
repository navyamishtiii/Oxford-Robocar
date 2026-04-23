import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import random

from src.preprocess import preprocess
from src.features import extract_features
from src.stats_analysis import run_statistical_tests
from src.ml_models import run_ml_models

# =========================
# PATH
# =========================
DATASET = Path("data/robodata/radar")

# =========================
# LOAD + SPLIT
# =========================
image_paths = sorted(DATASET.glob("*.png"))

split_idx = int(0.8 * len(image_paths))

train_paths = image_paths[:split_idx]
test_paths  = image_paths[split_idx:]

print(f"Train images: {len(train_paths)}")
print(f"Test images: {len(test_paths)}")

# =========================
# LOAD TRAIN DATA
# =========================
raw_images = []
for p in train_paths[:2000]:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        raw_images.append(img)

# =========================
# PREPROCESS
# =========================
images = preprocess(raw_images)

# =========================
# FEATURES
# =========================
features = extract_features(images)
df = pd.DataFrame(features)

# =========================
# 🔥 CORRECT LABELING (FIXED)
# =========================
entropy_thresh = df["spatial_entropy"].median()
density_thresh = df["reflection_density"].median()

df["label"] = (
    (df["spatial_entropy"] > entropy_thresh) &
    (df["reflection_density"] > density_thresh)
).astype(int)

df["label_name"] = df["label"].map({1: "Urban", 0: "Highway"})

print("\nBefore balancing:")
print(df["label_name"].value_counts())

# =========================
# BALANCE
# =========================
from sklearn.utils import resample

df_urban = df[df.label == 1]
df_highway = df[df.label == 0]

min_size = min(len(df_urban), len(df_highway))

df = pd.concat([
    resample(df_urban, n_samples=min_size, random_state=42),
    resample(df_highway, n_samples=min_size, random_state=42)
]).sample(frac=1).reset_index(drop=True)

# =========================
# 📊 MULTI-FEATURE COMPARISON
# =========================
features_to_plot = [
    "spatial_entropy",
    "reflection_density",
    "clutter_index",
    "temporal_variance",
    "mean_intensity",
    "std_intensity"
]

plt.figure(figsize=(14,10))

for i, feat in enumerate(features_to_plot):
    plt.subplot(3,2,i+1)
    df[df.label==1][feat].hist(alpha=0.5, label="Urban")
    df[df.label==0][feat].hist(alpha=0.5, label="Highway")
    plt.title(feat)
    plt.legend()

plt.tight_layout()
plt.show()

# =========================
# 📊 MEAN COMPARISON
# =========================
print("\n📊 FEATURE MEANS\n")
print(df.groupby("label_name")[features_to_plot].mean())

# =========================
# STATS + ML
# =========================
run_statistical_tests(df)
models, scaler = run_ml_models(df)

# =========================
# SAVE MODEL
# =========================
joblib.dump(models["Random Forest"], "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = preprocess([img])[0]

    feat = extract_features([img])[0]

    print("\nDEBUG FEATURES:", feat)  # helpful

    X = pd.DataFrame([{
        "mean_intensity": feat["mean_intensity"],
        "std_intensity": feat["std_intensity"],
        "skewness": feat["skewness"],
        "kurtosis": feat["kurtosis"],
        "clutter_index": feat["clutter_index"],
        "temporal_variance": feat["temporal_variance"]
    }])

    X_scaled = scaler.transform(X)

    model = models["Random Forest"]

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return ("Urban" if pred==1 else "Highway", max(prob))

# =========================
# TEST ON UNSEEN DATA
# =========================
sample = str(random.choice(test_paths))

label, conf = predict_image(sample)

# 🔥 Convert confidence to category
if conf > 0.85:
    level = "High"
elif conf > 0.65:
    level = "Medium"
else:
    level = "Low"

print(f"\n🔍 Prediction: {label} (Confidence: {level} - {conf:.2f})")
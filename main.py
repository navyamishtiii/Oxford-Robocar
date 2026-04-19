import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from src.preprocess import preprocess
from src.features import extract_features
from src.stats_analysis import run_statistical_tests
from src.ml_models import run_ml_models

# =========================
# 📁 PATH
# =========================
DATASET = Path("data/robodata/radar")

# =========================
# 📥 LOAD + SPLIT DATA
# =========================
image_paths = sorted(DATASET.glob("*.png"))

split_idx = int(0.8 * len(image_paths))

train_paths = image_paths[:split_idx]
test_paths  = image_paths[split_idx:]

print(f"Train images: {len(train_paths)}")
print(f"Test images: {len(test_paths)}")

# =========================
# LOAD TRAIN DATA ONLY
# =========================
raw_images = []
for p in train_paths[:2000]:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        raw_images.append(img)

print(f"Loaded {len(raw_images)} training images")

# =========================
# PREPROCESS
# =========================
images = preprocess(raw_images)

# =========================
# FEATURE EXTRACTION
# =========================
features = extract_features(images)
df = pd.DataFrame(features)

# =========================
# LABELING
# =========================
entropy_thresh = df["spatial_entropy"].median()
density_thresh = df["reflection_density"].median()

df["label"] = (
    (df["spatial_entropy"] > entropy_thresh).astype(int) +
    (df["reflection_density"] > density_thresh).astype(int)
)

df["label"] = (df["label"] >= 1).astype(int)
df["label_name"] = df["label"].map({1: "Urban", 0: "Highway"})

print("\nLabel distribution BEFORE balancing:")
print(df["label_name"].value_counts())

# =========================
# BALANCING
# =========================
from sklearn.utils import resample

df_urban = df[df.label == 1]
df_highway = df[df.label == 0]

min_size = min(len(df_urban), len(df_highway))

df_urban = resample(df_urban, replace=False, n_samples=min_size, random_state=42)
df_highway = resample(df_highway, replace=False, n_samples=min_size, random_state=42)

df = pd.concat([df_urban, df_highway]).sample(frac=1).reset_index(drop=True)

# =========================
# VISUALIZATION
# =========================
plt.figure()
df[df.label==1]["spatial_entropy"].hist(alpha=0.5, label="Urban")
df[df.label==0]["spatial_entropy"].hist(alpha=0.5, label="Highway")
plt.legend()
plt.title("Entropy Distribution")
plt.show()

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

print("✅ Model saved")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = preprocess([img])[0]

    feat = extract_features([img])[0]

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

    label = "Urban" if pred == 1 else "Highway"
    confidence = max(prob)

    return label, confidence

# =========================
# TEST ON UNSEEN DATA
# =========================
sample = str(test_paths[0])

label, conf = predict_image(sample)
print(f"\n🔍 Prediction: {label} (Confidence: {conf:.2f})")
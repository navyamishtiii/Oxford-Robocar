import numpy as np
from scipy.stats import skew, kurtosis, entropy


# =========================
# 🔥 FIXED ENTROPY FUNCTION
# =========================
def compute_entropy(img):
    # Proper float histogram (NO uint8 conversion)
    hist, _ = np.histogram(img.flatten(), bins=64, range=(0, 1), density=True)
    hist = hist + 1e-6  # avoid log(0)
    return entropy(hist)


# =========================
# 📡 REFLECTION DENSITY
# =========================
def reflection_density(img, threshold=0.5):
    return np.sum(img > threshold) / img.size


# =========================
# 🌪️ CLUTTER INDEX
# =========================
def clutter_index(img):
    return np.std(img)


# =========================
# ⏱️ TEMPORAL VARIANCE
# =========================
def temporal_variance(prev_img, curr_img):
    if prev_img is None:
        return 0
    return np.var(curr_img - prev_img)


# =========================
# 🧠 FEATURE EXTRACTION
# =========================
def extract_features(images):
    features = []
    prev = None

    for img in images:
        flat = img.flatten()

        feat = {
            "mean_intensity": np.mean(img),
            "std_intensity": np.std(img),
            "skewness": skew(flat),
            "kurtosis": kurtosis(flat),
            "spatial_entropy": compute_entropy(img),
            "clutter_index": clutter_index(img),
            "reflection_density": reflection_density(img),
            "temporal_variance": temporal_variance(prev, img)
        }

        features.append(feat)
        prev = img

    return features
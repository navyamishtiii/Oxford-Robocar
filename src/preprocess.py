import cv2
import numpy as np

def preprocess(images, size=(256, 256)):
    processed = []

    for img in images:
        img = cv2.resize(img, size)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)

        processed.append(img)

    return processed
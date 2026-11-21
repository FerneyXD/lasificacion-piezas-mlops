"""Ayudante para el monitoreo: extrae features simples de una imagen (no usa TensorFlow)."""
import numpy as np
from PIL import Image

FEATURES = ["width", "height", "mean_r", "mean_g", "mean_b", "brightness", "contrast", "sharpness"]


def image_features(img: Image.Image) -> dict:
    """Tamano, color medio, brillo, contraste y nitidez (varianza del laplaciano)."""
    w, h = img.size
    arr = np.asarray(img.convert("RGB"), dtype="float32")
    gray = arr.mean(axis=2)
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return {
        "width": w,
        "height": h,
        "mean_r": float(arr[..., 0].mean()),
        "mean_g": float(arr[..., 1].mean()),
        "mean_b": float(arr[..., 2].mean()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(lap.var()),
    }

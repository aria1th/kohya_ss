import numpy as np
import cv2
from cv2.ximgproc import guidedFilter

def cleanup_image(img: np.ndarray) -> np.ndarray:
    """
    Cleans up adversarial noise from an image.
    Reference : https://github.com/lllyasviel/AdverseCleaner/blob/main/clean.py
    """
    y = img.copy()

    for _ in range(64):
        y = cv2.bilateralFilter(y, 5, 8, 8)

    for _ in range(4):
        y = guidedFilter(img, y, 4, 16)

    return y.clip(0, 255).astype(np.uint8)

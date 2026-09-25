import cv2
import numpy as np
from PIL import Image


def generate_noiseprint_like_from_pil(pil_image: Image.Image) -> np.ndarray:
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    lap = np.abs(lap)
    lap = cv2.GaussianBlur(lap, (3, 3), 0)
    lap = (lap - lap.min()) / (lap.max() - lap.min() + 1e-8)
    return lap

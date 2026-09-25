"""Image loading helpers used by block V."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_image_gray(path: Path) -> Optional[np.ndarray]:
    """Return a 2D uint8 grayscale array, or None if the file is missing/unreadable."""
    if path is None or not Path(path).is_file():
        return None
    try:
        with Image.open(path) as im:
            if im.mode != "L":
                im = im.convert("L")
            arr = np.asarray(im, dtype=np.uint8)
        return arr
    except Exception:
        return None


def binarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Return a boolean foreground mask from a grayscale or RGB image."""
    if image is None:
        return np.zeros((0, 0), dtype=bool)
    img = np.asarray(image)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img >= threshold

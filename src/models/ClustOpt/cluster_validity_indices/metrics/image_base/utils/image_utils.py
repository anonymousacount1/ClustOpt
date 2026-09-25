# metrics/image/image_utils.py
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

@dataclass(frozen=True)
class RasterParams:
    grid_size: int = 256
    mode: Literal["binary", "density"] = "binary"
    blur_ksize: int = 3
    dilate_iters: int = 0

Bounds2D = Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)

def compute_global_bounds(X: np.ndarray, eps: float = 1e-9) -> Bounds2D:
    X = np.asarray(X)
    xmin = float(np.min(X[:, 0])); xmax = float(np.max(X[:, 0]))
    ymin = float(np.min(X[:, 1])); ymax = float(np.max(X[:, 1]))
    # avoid zero ranges
    if abs(xmax - xmin) < eps:
        xmax = xmin + eps
    if abs(ymax - ymin) < eps:
        ymax = ymin + eps
    return (xmin, xmax, ymin, ymax)

def rasterize_points(X: np.ndarray, params: RasterParams, bounds: Optional[Bounds2D] = None) -> np.ndarray:
    """
    Rasterize 2D points into an image grid.
    If bounds provided, uses global bounds for consistent mapping across clusters.
    """
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image metrics. Install opencv-python.")

    X = np.asarray(X)
    g = int(params.grid_size)

    if bounds is None:
        bounds = compute_global_bounds(X)
    xmin, xmax, ymin, ymax = bounds

    # normalize to [0, g-1]
    xs = (X[:, 0] - xmin) / (xmax - xmin)
    ys = (X[:, 1] - ymin) / (ymax - ymin)
    px = np.clip((xs * (g - 1)).astype(np.int32), 0, g - 1)
    py = np.clip(((1.0 - ys) * (g - 1)).astype(np.int32), 0, g - 1)  # invert y

    img = np.zeros((g, g), dtype=np.uint8)

    if params.mode == "binary":
        img[py, px] = 255
    else:
        # density: accumulate counts then scale to 0..255
        acc = np.zeros((g, g), dtype=np.float32)
        np.add.at(acc, (py, px), 1.0)
        if np.max(acc) > 0:
            acc = acc / np.max(acc) * 255.0
        img = acc.astype(np.uint8)

    if params.blur_ksize and params.blur_ksize >= 3:
        k = int(params.blur_ksize)
        if k % 2 == 0:
            k += 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    if params.dilate_iters and params.dilate_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=int(params.dilate_iters))

    return img

def canny_edges(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required for image metrics. Install opencv-python.")
    return cv2.Canny(img, threshold1=int(low), threshold2=int(high))

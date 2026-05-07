import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CropMeta:
    """Metadata to map pixel coordinates inside a resized crop back to frame coords.

    After calling crop_with_metadata(), any (cx, cy) point in the resized crop maps to:
        frame_x = roi_x1 + cx * scale_x
        frame_y = roi_y1 + cy * scale_y
    """
    roi_x1: int
    roi_y1: int    
    scale_x: float 
    scale_y: float 


def crop_with_metadata(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.4,
    target_size: int = 320,
) -> "Tuple[Optional[np.ndarray], Optional[CropMeta]]":
    """Crop a padded hand region and resize to target_size × target_size.

    padding=0.4 adds 40% of the bbox on each side → 1.8× total crop size.
    Returns (resized_crop, CropMeta) or (None, None) if the ROI is degenerate.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    roi_w = x2 - x1
    roi_h = y2 - y1

    if roi_w <= 0 or roi_h <= 0:
        return None, None

    crop_resized = cv2.resize(frame[y1:y2, x1:x2], (target_size, target_size))

    return crop_resized, CropMeta(
        roi_x1=x1,
        roi_y1=y1,
        scale_x=roi_w / target_size,
        scale_y=roi_h / target_size,
    )


def crop_hand_region(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.15,
) -> Optional[np.ndarray]:
    """Return a padded crop around bbox, or None."""
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    px, py = int(bw * padding), int(bh * padding)

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w, x + bw + px)
    y2 = min(h, y + bh + py)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    padding: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """Return bbox expanded by padding fraction, clamped to frame size."""
    x, y, bw, bh = bbox
    px, py = int(bw * padding), int(bh * padding)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(frame_w, x + bw + px)
    y2 = min(frame_h, y + bh + py)
    return x1, y1, x2 - x1, y2 - y1


def bboxes_overlap(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    min_overlap: float = 0.20,
) -> bool:
    """True when the intersection covers at least min_overlap of the smaller box.

    Uses intersection-over-min-area so that a phone held in a
    hand (where the phone bbox extends beyond the hand bbox) still registers as
    overlapping even when the two boxes differ greatly in size.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)

    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    min_area = min(w1 * h1, w2 * h2)
    return (intersection / min_area) >= min_overlap if min_area > 0 else False

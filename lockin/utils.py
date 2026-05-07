import cv2
import numpy as np
from typing import Optional, Tuple


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

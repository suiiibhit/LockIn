import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import crop_with_metadata

logger = logging.getLogger(__name__)

_PHONE_CLASS_ID      = 67
_FALLBACK_CONFIDENCE = 0.55


@dataclass
class PhoneDetection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    in_hand: bool


class PhoneClassifier:
    """
    Two-path phone detection:

    PRIMARY — ROI crops (fast, precise):
        Crops each detected hand at 1.8× its bbox, resizes to 320×320, and
        batches all crops in a single YOLO call.  All hits are in_hand=True by
        construction.  Faster than full-frame and eliminates desk FPs.

    FALLBACK — full-frame YOLO (belt-and-suspenders):
        Runs only when ROI path returns nothing AND ≥1 hand is already visible.
        This catches the common failure mode where the gripping hand itself is
        not detected by MediaPipe (occluded by the phone, tight grip, edge of
        frame).  Uses _FALLBACK_CONFIDENCE (0.55) to reduce table FPs.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        model_path: Optional[Path] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[Path]):
        try:
            from ultralytics import YOLO

            if model_path and Path(model_path).exists():
                self._model = YOLO(str(model_path))
                logger.info("YOLO model loaded from %s", model_path)
            else:
                model_name = Path(model_path).name if model_path else "yolov8s.pt"
                self._model = YOLO(model_name)
                logger.info("%s loaded (auto-downloaded if needed).", model_name)
        except ImportError:
            logger.error("ultralytics package not found. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise

    def classify(
        self,
        frame: np.ndarray,
        hand_bboxes: List[Tuple[int, int, int, int]],
    ) -> List[PhoneDetection]:
        """
        Detect phones using ROI crops, falling back to full-frame when needed.

        hand_bboxes must be non-empty (caller's responsibility — main.py already
        guards this).  Returns [] if model is unavailable.
        """
        if self._model is None or not hand_bboxes:
            return []

        detections = self._classify_roi(frame, hand_bboxes)

        if not detections:
            detections = self._classify_full_frame_fallback(frame)

        return detections

    # ------------------------------------------------------------------
    # Internal detection paths
    # ------------------------------------------------------------------

    def _classify_roi(
        self,
        frame: np.ndarray,
        hand_bboxes: List[Tuple[int, int, int, int]],
    ) -> List[PhoneDetection]:
        """Run YOLO on 320×320 hand crops (primary path)."""
        fh, fw = frame.shape[:2]

        crops, metas = [], []
        for hbbox in hand_bboxes:
            crop, meta = crop_with_metadata(frame, hbbox)
            if crop is not None:
                crops.append(crop)
                metas.append(meta)

        if not crops:
            return []

        results = self._model(crops, verbose=False)

        detections: List[PhoneDetection] = []
        for result, meta in zip(results, metas):
            for box in result.boxes:
                if int(box.cls[0]) != _PHONE_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                cx1, cy1, cx2, cy2 = map(float, box.xyxy[0])
                fx1 = int(meta.roi_x1 + cx1 * meta.scale_x)
                fy1 = int(meta.roi_y1 + cy1 * meta.scale_y)
                fx2 = int(meta.roi_x1 + cx2 * meta.scale_x)
                fy2 = int(meta.roi_y1 + cy2 * meta.scale_y)

                fx1 = max(0, min(fw, fx1))
                fy1 = max(0, min(fh, fy1))
                fx2 = max(0, min(fw, fx2))
                fy2 = max(0, min(fh, fy2))

                detections.append(PhoneDetection(
                    bbox=(fx1, fy1, fx2 - fx1, fy2 - fy1),
                    confidence=conf,
                    in_hand=True,
                ))

        return detections

    def _classify_full_frame_fallback(
        self,
        frame: np.ndarray,
    ) -> List[PhoneDetection]:
        """Run YOLO on the full frame (fallback path).

        Only called when ROI found nothing AND ≥1 hand is in frame.
        Higher confidence threshold (_FALLBACK_CONFIDENCE) reduces desk FPs.
        """
        fh, fw = frame.shape[:2]
        results = self._model(frame, verbose=False)[0]

        detections: List[PhoneDetection] = []
        for box in results.boxes:
            if int(box.cls[0]) != _PHONE_CLASS_ID:
                continue
            conf = float(box.conf[0])
            if conf < _FALLBACK_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = max(0, min(fw, x1))
            y1 = max(0, min(fh, y1))
            x2 = max(0, min(fw, x2))
            y2 = max(0, min(fh, y2))

            phone_bbox = (x1, y1, x2 - x1, y2 - y1)
            detections.append(PhoneDetection(
                bbox=phone_bbox,
                confidence=conf,
                in_hand=True,
            ))
            logger.debug(
                "Fallback full-frame hit: conf=%.2f  bbox=%s", conf, phone_bbox
            )

        return detections

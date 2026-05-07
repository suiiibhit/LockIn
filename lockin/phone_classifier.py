import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .utils import bboxes_overlap, expand_bbox

logger = logging.getLogger(__name__)

_PHONE_CLASS_ID = 67


@dataclass
class PhoneDetection:
    bbox: Tuple[int, int, int, int]  
    confidence: float
    in_hand: bool


class PhoneClassifier:
    """
    Runs YOLOv8 on the full frame to find phones, then cross-references the
    detected bounding boxes with the hand regions supplied by HandDetector.
    Only phones that overlap a hand are treated as active distractions.
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
                logger.info(f"YOLO model loaded from {model_path}")
            else:
                model_name = Path(model_path).name if model_path else "yolov8s.pt"
                self._model = YOLO(model_name)
                logger.info(f"{model_name} loaded (auto-downloaded if needed).")
        except ImportError:
            logger.error("ultralytics package not found. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            raise

    def classify(
        self,
        frame: np.ndarray,
        hand_bboxes: List[Tuple[int, int, int, int]],
    ) -> List[PhoneDetection]:
        """
        Detect phones in frame and mark those that overlap *hand_bboxes*.
        Returns an empty list when the model is unavailable.
        """
        if self._model is None:
            return []

        fh, fw = frame.shape[:2]
        results = self._model(frame, verbose=False)[0]
        detections: List[PhoneDetection] = []

        expanded_hand_bboxes = [
            expand_bbox(hbbox, 0.30, fw, fh) for hbbox in hand_bboxes
        ]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != _PHONE_CLASS_ID or conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phone_bbox = (x1, y1, x2 - x1, y2 - y1)

            in_hand = any(
                bboxes_overlap(phone_bbox, hbbox) for hbbox in expanded_hand_bboxes
            )

            detections.append(
                PhoneDetection(bbox=phone_bbox, confidence=conf, in_hand=in_hand)
            )

        return detections

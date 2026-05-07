import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb finger
    (0, 5), (5, 6), (6, 7), (7, 8),            # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky finger
    (0, 17),                                   # base of palm
]


@dataclass
class HandDetection:
    landmarks: list                       
    bbox: Tuple[int, int, int, int]          
    handedness: str                          
    confidence: float


class HandDetector:
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        model_path: Path = _DEFAULT_MODEL_PATH,
    ):
        self._model_path = model_path
        self._ensure_model_downloaded()

        base_options = mp_python.BaseOptions(
            model_asset_path=str(self._model_path)
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.50, 
            min_hand_presence_confidence=0.40,   
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = 0
        logger.info("HandDetector initialised (MediaPipe Tasks API).")

    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        """Return a HandDetection for every hand visible in *frame*."""
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts

        result = self._detector.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return []

        detections: List[HandDetection] = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            score = result.handedness[i][0].score

            xs = [lm.x * w for lm in hand_lms]
            ys = [lm.y * h for lm in hand_lms]
            bbox = (
                int(min(xs)),
                int(min(ys)),
                int(max(xs)) - int(min(xs)),
                int(max(ys)) - int(min(ys)),
            )

            detections.append(
                HandDetection(
                    landmarks=hand_lms,
                    bbox=bbox,
                    handedness=label,
                    confidence=score,
                )
            )
        return detections

    def draw_landmarks(
        self, frame: np.ndarray, detections: List[HandDetection]
    ) -> np.ndarray:
        """Draw skeleton overlay for all detected hands."""
        h, w = frame.shape[:2]
        for det in detections:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in det.landmarks]

            for start, end in _HAND_CONNECTIONS:
                cv2.line(frame, pts[start], pts[end], (0, 200, 0), 2, cv2.LINE_AA)

            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, pt, 4, (255, 255, 255), 1)

        return frame

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _ensure_model_downloaded(self):
        if self._model_path.exists():
            return
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Hand landmarker model not found — downloading (~9 MB) from Google…"
        )
        try:
            urllib.request.urlretrieve(_MODEL_URL, str(self._model_path))
            logger.info("Model saved to %s", self._model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download hand landmarker model: {exc}\n"
                f"Download it manually from:\n  {_MODEL_URL}\n"
                f"and place it at: {self._model_path}"
            ) from exc

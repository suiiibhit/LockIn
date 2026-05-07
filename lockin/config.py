"""
Configuration for LockIn!
All tuneable defaults live here so they are easy to find and change.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
SOUNDS_DIR   = PROJECT_ROOT / "sounds"
MODELS_DIR   = PROJECT_ROOT / "models"

PHONE_DETECTED_SOUND      = SOUNDS_DIR / "phone_detected.wav"
PHONE_DETECTED_LOOP_SOUND = SOUNDS_DIR / "phone_detected_loop.wav"
GOOD_BOY_SOUND            = SOUNDS_DIR / "good_boy.wav"

YOLO_MODEL = MODELS_DIR / "yolov8s.pt"   # auto-downloaded on first run if absent

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH         = 640
CAMERA_HEIGHT        = 480

# ---------------------------------------------------------------------------
# Hand detection: MediaPipe
# ---------------------------------------------------------------------------
MAX_HANDS                 = 2
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE  = 0.5

# ---------------------------------------------------------------------------
# Phone detection: YOLOv8
# ---------------------------------------------------------------------------
PHONE_DETECTION_CONFIDENCE = 0.20

# ---------------------------------------------------------------------------
# Alert state-machine parameters
# ---------------------------------------------------------------------------
ALERT_SMOOTHING_FRAMES  = 3    # consecutive frames needed to trigger alert
ALERT_RESET_FRAMES      = 15   # consecutive frames without phone to dismiss alert
ALERT_VOLUME            = 80   # 0–100
GOOD_BOY_MIN_FRAMES     = 45   # minimum frames to show "Good Boy!" (~1.5 s at 30 FPS)

# ---------------------------------------------------------------------------
# Logging information
# ---------------------------------------------------------------------------
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

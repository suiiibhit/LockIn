"""
Configuration for LockIn!
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directories
SOUNDS_DIR = PROJECT_ROOT / "sounds"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Audio files
PHONE_DETECTED_SOUND = SOUNDS_DIR / "phone_detected.wav"
PHONE_DETECTED_LOOP_SOUND = SOUNDS_DIR / "phone_detected_loop.wav"
GOOD_BOY_SOUND = SOUNDS_DIR / "good_boy.wav"

# Camera settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# Hand detection settings (MediaPipe)
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 2

# Phone detection settings
PHONE_DETECTION_CONFIDENCE = 0.5
PHONE_CLASSIFICATION_TIMEOUT = 40  # milliseconds

# Alert settings
ALERT_SMOOTHING_FRAMES = 3  # Require 3+ consecutive frames before alerting
ALERT_RESET_FRAMES = 2  # Reset counter if phone not detected for 2+ frames
ALERT_VOLUME = 80  # 0-100

# UI settings
ALERT_COLOR = (0, 0, 255)  # BGR: Red
BOUNDING_BOX_COLOR = (0, 0, 255)  # BGR: Red
BOUNDING_BOX_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  # BGR: Red
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# Performance settings
ENABLE_GPU = True  # Auto-detect GPU if available
SKIP_FRAMES_FOR_INFERENCE = 1  # Process every N frames

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model paths
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"  # Will auto-download if missing

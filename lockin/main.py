"""
LockIn!'s entry point.
Run with:  python -m lockin (after installing dependencies and setting up sounds (if needed))
"""

import argparse
import logging
import sys

import cv2

from . import config
from .alert_system import AlertSystem
from .camera import Camera
from .hand_detector import HandDetector
from .phone_classifier import PhoneClassifier
from .tracker import PhoneTracker
from .ui import draw_alert_overlay, draw_controls_hud, draw_phone_bboxes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI (Command Line Interface)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lockin",
        description="LockIn! — stay off your phone, stay in the zone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Controls while running:\n"
            "  Q / ESC  quit\n"
            "  M        toggle mute\n"
            "  D        toggle debug overlay\n"
        ),
    )
    p.add_argument("--camera",     type=int,   default=config.DEFAULT_CAMERA_INDEX,
                   help="camera index (default: 0)")
    p.add_argument("--confidence", type=float, default=config.PHONE_DETECTION_CONFIDENCE,
                   help="phone detection confidence threshold 0-1 (default: 0.5)")
    p.add_argument("--volume",     type=int,   default=config.ALERT_VOLUME,
                   help="alert volume 0-100 (default: 80)")
    p.add_argument("--width",      type=int,   default=config.CAMERA_WIDTH)
    p.add_argument("--height",     type=int,   default=config.CAMERA_HEIGHT)
    p.add_argument("--log-level",  default=config.LOG_LEVEL,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-landmarks", action="store_true",
                   help="hide hand skeleton overlay")
    return p.parse_args()


def _setup_logging(level: str):
    logging.basicConfig(level=getattr(logging, level, logging.INFO),
                        format=config.LOG_FORMAT)


def _warn_missing_audio():
    missing = [
        p.name for p in (
            config.PHONE_DETECTED_SOUND,
            config.PHONE_DETECTED_LOOP_SOUND,
            config.GOOD_BOY_SOUND,
        )
        if not p.exists()
    ]
    if missing:
        logger.warning(
            "Audio files not found in sounds/: %s. "
            "Alerts will be visual only. Add WAV files to enable audio.",
            ", ".join(missing),
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace):
    logger.info("LockIn! starting up…")
    _warn_missing_audio()

    show_landmarks = not args.no_landmarks
    show_debug     = False

    with Camera(args.camera, args.width, args.height) as camera, \
         HandDetector(config.MAX_HANDS,
                      config.HAND_DETECTION_CONFIDENCE,
                      config.HAND_TRACKING_CONFIDENCE) as hand_detector:

        phone_classifier = PhoneClassifier(
            confidence_threshold=args.confidence,
            model_path=config.YOLO_MODEL,
        )

        alert_system = AlertSystem(
            phone_detected_sound=config.PHONE_DETECTED_SOUND,
            phone_loop_sound=config.PHONE_DETECTED_LOOP_SOUND,
            good_boy_sound=config.GOOD_BOY_SOUND,
            smoothing_frames=config.ALERT_SMOOTHING_FRAMES,
            reset_frames=config.ALERT_RESET_FRAMES,
            volume=args.volume / 100.0,
        )

        phone_tracker = PhoneTracker(max_age=8, iou_thresh=0.25)

        logger.info("All systems go. Press Q or ESC to quit.")

        frame_n          = 0
        hand_detections  = []
        phone_detections = []
        tracked_phones   = []

        try:
            while camera.is_running:
                frame = camera.read_frame()
                if frame is None:
                    continue

                frame_n += 1

                hand_detections = hand_detector.detect(frame)

                if hand_detections:
                    if frame_n % 2 == 0:
                        hand_bboxes      = [d.bbox for d in hand_detections]
                        phone_detections = phone_classifier.classify(frame, hand_bboxes)
                    else:
                        phone_detections = []   # tracker will interpolate
                    tracked_phones = phone_tracker.update(phone_detections)
                else:
                    phone_detections = []
                    tracked_phones   = phone_tracker.update([])  # age all tracks

                phone_in_hand = len(tracked_phones) > 0

                # State machine
                alert_state = alert_system.update(phone_in_hand)

                # Render
                if show_landmarks:
                    frame = hand_detector.draw_landmarks(frame, hand_detections)

                frame = draw_phone_bboxes(frame, tracked_phones)
                frame = draw_alert_overlay(frame, alert_state, alert_system.is_muted)
                frame = draw_controls_hud(frame)

                if show_debug:
                    _draw_debug(frame, hand_detections, phone_detections,
                                tracked_phones, alert_state, frame_n)

                cv2.imshow("LockIn!", frame)

                key = cv2.waitKey(1) & 0xFF
                # Handle keyboard controls
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty("LockIn!", cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key in (ord("m"), ord("M")):
                    alert_system.toggle_mute()
                elif key in (ord("d"), ord("D")):
                    show_debug = not show_debug

        finally:
            alert_system.cleanup()
            cv2.destroyAllWindows()
            logger.info("LockIn! stopped.")


# ---------------------------------------------------------------------------
# Debug overlay and main entry point
# ---------------------------------------------------------------------------

def _draw_debug(frame, hands, phones, tracked, state, frame_n):
    lines = [
        f"frame  : {frame_n}",
        f"hands  : {len(hands)}",
        f"yolo   : {len(phones)}  in-hand={sum(1 for p in phones if p.in_hand)}",
        f"tracks : {len(tracked)}  (ages: {[t.age for t in tracked]})",
        f"state  : {state.name}",
    ]
    y = 18
    for line in lines:
        cv2.putText(frame, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 18

def main():
    args = _parse_args()
    _setup_logging(args.log_level)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except RuntimeError as exc:
        logger.error(exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

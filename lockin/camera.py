import cv2
import logging

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self._cap = None
        self._running = False

    def start(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.camera_index}. "
                "Check that your webcam is connected and not in use by another app."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._running = True
        logger.info(f"Camera started (index={self.camera_index}, {self.width}x{self.height})")

    def read_frame(self):
        if not self._running:
            return None
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            return None
        # Flip horizontally so the feed feels like a mirror (front-camera UX)
        return cv2.flip(frame, 1)

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            logger.info("Camera released.")

    @property
    def is_running(self) -> bool:
        return self._running and self._cap is not None and self._cap.isOpened()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

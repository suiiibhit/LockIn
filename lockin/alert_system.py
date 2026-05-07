import logging
import threading
from enum import Enum, auto
from pathlib import Path
from typing import Optional

_GOOD_BOY_MIN_FRAMES = 45 

logger = logging.getLogger(__name__)


class AlertState(Enum):
    IDLE = auto()
    ALERTING = auto()
    GOOD_BOY = auto()


class AlertSystem:
    """
    Owns all audio playback and the phone-detection state machine.

    State transitions:
        IDLE      → ALERTING   : phone detected for `smoothing_frames` consecutive frames
        ALERTING  → GOOD_BOY   : phone absent for `reset_frames` consecutive frames
        GOOD_BOY  → ALERTING   : phone re-detected for `smoothing_frames` frames
        GOOD_BOY  → IDLE       : overlay shown for _GOOD_BOY_MIN_FRAMES with no phone
    """

    def __init__(
        self,
        phone_detected_sound: Path,
        phone_loop_sound: Path,
        good_boy_sound: Path,
        smoothing_frames: int = 3,
        reset_frames: int = 2,
        volume: float = 0.8,
    ):
        self._sounds_cfg = {
            "detected": phone_detected_sound,
            "loop": phone_loop_sound,
            "good_boy": good_boy_sound,
        }
        self.smoothing_frames = smoothing_frames
        self.reset_frames = reset_frames
        self.volume = max(0.0, min(1.0, volume))

        self.state = AlertState.IDLE
        self._detect_streak   = 0
        self._absent_streak   = 0
        self._good_boy_frames = 0   
        self._loop_started    = False
        self._muted = False
        self._lock = threading.Lock()

        self._audio_ok = False
        self._sounds: dict = {}
        self._init_audio()

    def update(self, phone_in_hand: bool) -> AlertState:
        """Call once per frame with the current detection result."""
        with self._lock:
            if phone_in_hand:
                self._detect_streak += 1
                self._absent_streak = 0
            else:
                self._absent_streak += 1
                self._detect_streak = 0

            if self.state is AlertState.IDLE:
                if self._detect_streak >= self.smoothing_frames:
                    self._enter_alerting()

            elif self.state is AlertState.ALERTING:
                if self._absent_streak >= self.reset_frames:
                    self._enter_good_boy()
                elif not self._muted and not self._loop_started and not self._is_audio_playing():
                    self._loop_started = True
                    self._play_loop()

            elif self.state is AlertState.GOOD_BOY:
                self._good_boy_frames += 1
                if self._detect_streak >= self.smoothing_frames:
                    self._enter_alerting()
                elif self._good_boy_frames >= _GOOD_BOY_MIN_FRAMES:
                    self._enter_idle()

            return self.state

    def toggle_mute(self) -> bool:
        with self._lock:
            self._muted = not self._muted
            if self._muted:
                self._stop_all()
            elif self.state is AlertState.ALERTING:
                self._loop_started = True
                self._play_loop()
        logger.info("Audio %s.", "muted" if self._muted else "unmuted")
        return self._muted

    @property
    def is_muted(self) -> bool:
        return self._muted

    def cleanup(self):
        self._stop_all()
        if self._audio_ok:
            import pygame
            pygame.mixer.quit()

    # ------------------------------------------------------------------
    # Transitions 
    # ------------------------------------------------------------------

    def _enter_alerting(self):
        self.state = AlertState.ALERTING
        self._loop_started = False
        self._stop_all()
        if not self._muted:
            self._play_once("detected")
        logger.debug("Alert state → ALERTING")

    def _enter_good_boy(self):
        self.state = AlertState.GOOD_BOY
        self._good_boy_frames = 0
        self._stop_all()
        if not self._muted:
            self._play_once("good_boy")
        logger.debug("Alert state → GOOD_BOY")

    def _enter_idle(self):
        self.state = AlertState.IDLE
        self._detect_streak = 0
        self._absent_streak = 0
        logger.debug("Alert state → IDLE")

    # ------------------------------------------------------------------
    # Audio helper functions
    # ------------------------------------------------------------------

    def _init_audio(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self._audio_ok = True

            for name, path in self._sounds_cfg.items():
                if path.exists():
                    snd = pygame.mixer.Sound(str(path))
                    snd.set_volume(self.volume)
                    self._sounds[name] = snd
                else:
                    logger.warning("Audio file missing: %s  (visual alerts still active)", path)
                    self._sounds[name] = None

            logger.info("Audio system ready.")
        except Exception as exc:
            logger.warning("Audio init failed (%s). Running in visual-only mode.", exc)
            self._audio_ok = False

    def _play_loop(self):
        snd = self._sounds.get("loop")
        if snd:
            snd.play(loops=-1)

    def _play_once(self, key: str):
        snd = self._sounds.get(key)
        if snd:
            snd.play()

    def _stop_all(self):
        if self._audio_ok:
            import pygame
            pygame.mixer.stop()

    def _is_audio_playing(self) -> bool:
        if not self._audio_ok:
            return False
        import pygame
        return pygame.mixer.get_busy()

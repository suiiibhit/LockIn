import cv2
import numpy as np
from typing import List

from .alert_system import AlertState
from .hand_detector import HandDetection
from .phone_classifier import PhoneDetection
from .utils import bboxes_overlap

_RED    = (0,   0,   255)
_GREEN  = (0,   255, 0  )
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0  )
_ORANGE = (0,   165, 255)
_DARK   = (20,  20,  20 )
_GREY   = (160, 160, 160)

_CONTROLS = [
    ("Q / ESC", "Quit"),
    ("M",       "Mute toggle"),
    ("D",       "Debug info"),
]


# ---------------------------------------------------------------------------
# object (hand + phone) drawing helpers
# ---------------------------------------------------------------------------

def draw_hand_bboxes(
    frame: np.ndarray,
    hand_detections: List[HandDetection],
    phone_detections: List[PhoneDetection],
    alert_state: AlertState,
) -> np.ndarray:
    """Outline each hand. Red when a phone overlaps it, Green otherwise."""
    for hand in hand_detections:
        has_phone = any(
            pd.in_hand and bboxes_overlap(hand.bbox, pd.bbox)
            for pd in phone_detections
        )
        color = _RED if (has_phone or alert_state is AlertState.ALERTING) else _GREEN
        x, y, w, h = hand.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2 + has_phone)
    return frame


def draw_phone_bboxes(
    frame: np.ndarray,
    phone_detections: List[PhoneDetection],
) -> np.ndarray:
    """Draw a labelled Red box around every detected phone."""
    for phone in phone_detections:
        x, y, w, h = phone.bbox
        label = f"Phone {phone.confidence:.0%}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), _RED, 2)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - lh - 8), (x + lw + 6, y), _RED, cv2.FILLED)
        cv2.putText(
            frame, label, (x + 3, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA,
        )
    return frame


# ---------------------------------------------------------------------------
# Frame alert overlays
# ---------------------------------------------------------------------------

def draw_alert_overlay(
    frame: np.ndarray,
    alert_state: AlertState,
    is_muted: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    if alert_state is AlertState.ALERTING:
        _draw_border(frame, w, h, _RED, alpha=0.55)
        _draw_centred_text(frame, w, h, "LOCK IN!", _RED, scale=3.0, thickness=4)
        _draw_subtext(frame, w, h, "Put your phone down!", offset_y=50)

    elif alert_state is AlertState.GOOD_BOY:
        _draw_border(frame, w, h, _GREEN, alpha=0.30)
        _draw_centred_text(frame, w, h, "GOOD BOY!", _GREEN, scale=3.0, thickness=4)
        _draw_subtext(frame, w, h, "Stay focused!", offset_y=50)

    if is_muted:
        cv2.putText(
            frame, "MUTED", (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, _ORANGE, 2, cv2.LINE_AA,
        )

    return frame


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def _draw_border(
    frame: np.ndarray,
    w: int,
    h: int,
    color: tuple,
    alpha: float = 0.5,
    thickness: int = 22,
):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, thickness)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_centred_text(
    frame: np.ndarray,
    w: int,
    h: int,
    text: str,
    color: tuple,
    scale: float = 3.0,
    thickness: int = 4,
    font=cv2.FONT_HERSHEY_DUPLEX,
):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = (w - tw) // 2
    ty = (h + th) // 2

    cv2.putText(frame, text, (tx + 3, ty + 3), font,
                scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (tx, ty), font,
                scale, color, thickness, cv2.LINE_AA)

    return ty, th


def _draw_subtext(
    frame: np.ndarray,
    w: int,
    h: int,
    text: str,
    offset_y: int = 50,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.9,
    thickness: int = 2,
):
    (sw, sh), _ = cv2.getTextSize(text, font, scale, thickness)
    sx = (w - sw) // 2
    sy = h // 2 + sh + offset_y + 20
    cv2.putText(frame, text, (sx, sy), font, scale, _WHITE, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Keyboard controls HUD
# ---------------------------------------------------------------------------

def draw_controls_hud(frame: np.ndarray) -> np.ndarray:
    """Render a polished semi-transparent controls panel in the bottom-right."""
    fh, fw = frame.shape[:2]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    key_scale  = 0.42
    val_scale  = 0.38
    thickness  = 1
    line_h     = 18
    pad_x      = 10
    pad_y      = 8
    corner_r   = 6

    key_w = max(cv2.getTextSize(k, font, key_scale, thickness)[0][0] for k, _ in _CONTROLS)
    val_w = max(cv2.getTextSize(v, font, val_scale, thickness)[0][0] for _, v in _CONTROLS)
    sep   = 8

    panel_w = pad_x * 2 + key_w + sep + val_w
    panel_h = pad_y * 2 + line_h * len(_CONTROLS) + 4

    margin  = 12
    x0 = fw - panel_w - margin
    y0 = fh - panel_h - margin
    x1 = fw - margin
    y1 = fh - margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)

    header_y = y0 + pad_y - 3
    cv2.line(frame,
             (x0 + 4, header_y + line_h + 2),
             (x1 - 4, header_y + line_h + 2),
             (60, 60, 60), 1)

    for i, (key, val) in enumerate(_CONTROLS):
        row_y = y0 + pad_y + i * line_h + line_h - 3

        (kw, kh), _ = cv2.getTextSize(key, font, key_scale, thickness)
        bx0 = x0 + pad_x - 3
        bx1 = x0 + pad_x + kw + 3
        by0 = row_y - kh - 1
        by1 = row_y + 3
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (90, 90, 90), 1)

        cv2.putText(frame, key,
                    (x0 + pad_x, row_y),
                    font, key_scale, _WHITE, thickness, cv2.LINE_AA)

        dot_x = x0 + pad_x + key_w + sep // 2
        cv2.circle(frame, (dot_x, row_y - kh // 2), 1, _GREY, cv2.FILLED)

        cv2.putText(frame, val,
                    (x0 + pad_x + key_w + sep, row_y),
                    font, val_scale, _GREY, thickness, cv2.LINE_AA)

    return frame

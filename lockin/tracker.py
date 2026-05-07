"""
Lightweight IoU-based phone tracker.

Purpose
-------
YOLO confidence wobbles frame-to-frame. Without a tracker, a phone held in
hand produces flickering detections: detect / miss / detect / miss …  The
AlertSystem's 3-frame streak never completes, so LOCK IN! never fires.

The tracker bridges these dropouts. A phone detected in frame N is kept alive
for up to `max_age` frames even if YOLO misses it.  Downstream code sees a
stable, continuous detection signal instead of noise.

Usage
-----
    tracker = PhoneTracker()
    tracked = tracker.update(yolo_detections)   # call every frame
    phone_in_hand = len(tracked) > 0
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrackedPhone:
    """A phone detection stabilised across frames by the tracker.

    Intentionally mirrors PhoneDetection's public attributes (.bbox,
    .confidence, .in_hand) so it can be passed directly to UI helpers.
    """
    track_id:   int
    bbox:       Tuple[int, int, int, int]
    confidence: float
    in_hand:    bool = True
    age:        int  = 0


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """Intersection-over-union for (x, y, w, h) bounding boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class PhoneTracker:
    """
    Greedy IoU tracker for phone detections.

    Each frame:
      1. Match incoming detections to existing tracks via IoU.
      2. Update matched tracks with the new bbox / confidence; reset age = 0.
      3. Age unmatched tracks by 1; their bbox stays where it was (phones move
         slowly relative to frame rate, so static extrapolation is fine).
      4. Spawn a new track for each unmatched detection.
      5. Prune tracks whose age exceeds max_age.

    Result: a smooth list of living tracks that persists across short YOLO gaps.
    """

    def __init__(self, max_age: int = 8, iou_thresh: float = 0.25):
        """
        max_age    : frames a track survives without a matching detection.
                     8 frames ≈ 0.27 s at 30 FPS — enough to bridge normal
                     YOLO dropout bursts without introducing long ghost tracks.
        iou_thresh : minimum IoU to consider a detection a match for a track.
        """
        self.max_age    = max_age
        self.iou_thresh = iou_thresh
        self._tracks: List[TrackedPhone] = []
        self._next_id   = 0

    def update(self, detections: List) -> List[TrackedPhone]:
        """
        Match *detections* (List[PhoneDetection]) to existing tracks.

        Passing an empty list is valid — all tracks age by 1 (used when phone
        detection is intentionally skipped to save compute).

        Returns all living tracks (age ≤ max_age).
        """
        matched_det = set()
        matched_trk = set()

        if self._tracks and detections:
            flat = [
                (_iou(self._tracks[ti].bbox, detections[di].bbox), ti, di)
                for ti in range(len(self._tracks))
                for di in range(len(detections))
            ]
            flat.sort(key=lambda x: x[0], reverse=True)

            for score, ti, di in flat:
                if score < self.iou_thresh:
                    break
                if ti in matched_trk or di in matched_det:
                    continue
                trk = self._tracks[ti]
                trk.bbox       = detections[di].bbox
                trk.confidence = detections[di].confidence
                trk.age        = 0
                matched_trk.add(ti)
                matched_det.add(di)

        for ti, trk in enumerate(self._tracks):
            if ti not in matched_trk:
                trk.age += 1

        for di, det in enumerate(detections):
            if di not in matched_det:
                self._tracks.append(TrackedPhone(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    in_hand=det.in_hand,
                    age=0,
                ))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        if self._tracks or detections:
            logger.debug(
                "Tracker: %d det → %d live tracks  (ids: %s)",
                len(detections),
                len(self._tracks),
                [t.track_id for t in self._tracks],
            )

        return list(self._tracks)

    def reset(self):
        """Clear all tracks (e.g., on camera reinit)."""
        self._tracks.clear()

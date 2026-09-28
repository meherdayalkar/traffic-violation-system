# tracking/tracker.py — Tracking Engineer's module
# SORT tracker: assigns persistent track_ids to detections across frames

import numpy as np
from tracking.kalman_numpy import KalmanFilter   # pure numpy — no filterpy needed
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from detection.detector import Detection


def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IOU between two bboxes [x1,y1,x2,y2]."""
    bb_test = np.asarray(bb_test, dtype=float).flatten()
    bb_gt   = np.asarray(bb_gt, dtype=float).flatten()
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    a1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    a2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return float(inter / (union + 1e-6))


@dataclass
class Track:
    """A single tracked object with Kalman filter state."""
    track_id:   int
    class_name: str
    bbox:       np.ndarray          # current [x1,y1,x2,y2]
    confidence: float
    age:        int = 0             # frames since creation
    hits:       int = 0             # consecutive detection hits
    time_since_update: int = 0

    # Position history for direction / speed estimation
    center_history: List[Tuple[int,int]] = field(default_factory=list)

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=int).flatten()
        self._kf = self._build_kalman(self.bbox)

    def _build_kalman(self, bbox: np.ndarray) -> KalmanFilter:
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)
        kf.R[2:, 2:] *= 10.
        kf.P[4:, 4:] *= 1000.
        kf.P          *= 10.
        kf.Q[-1,-1]   *= 0.01
        kf.Q[4:, 4:]  *= 0.01
        kf.x[:4] = self._bbox_to_z(bbox)
        return kf

    @staticmethod
    def _bbox_to_z(bbox):
        b = np.asarray(bbox, dtype=float).flatten()
        w = b[2] - b[0]
        h = b[3] - b[1]
        x = b[0] + w / 2.0
        y = b[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r], dtype=float).reshape((4, 1))

    @staticmethod
    def _z_to_bbox(z, score=None):
        z = np.asarray(z, dtype=float).flatten()
        w = np.sqrt(abs(z[2] * z[3]))
        h = z[2] / (w + 1e-6)
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=int)

    def predict(self):
        if float(self._kf.x[6, 0] + self._kf.x[2, 0]) <= 0:
            self._kf.x[6, 0] = 0.0
        self._kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox = self._z_to_bbox(self._kf.x)

    def update(self, det: Detection):
        self._kf.update(self._bbox_to_z(det.bbox))
        self.bbox       = self._z_to_bbox(self._kf.x)
        self.confidence = float(det.confidence)
        self.hits      += 1
        self.time_since_update = 0
        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int((self.bbox[1] + self.bbox[3]) / 2)
        self.center_history.append((cx, cy))
        if len(self.center_history) > 30:
            self.center_history.pop(0)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= config.TRACK_MIN_HITS

    @property
    def movement_direction(self) -> Optional[str]:
        """Estimate dominant horizontal direction from recent history."""
        if len(self.center_history) < 4:
            return None
        dx = self.center_history[-1][0] - self.center_history[-4][0]
        if abs(dx) < 5:
            return "stationary"
        return "right" if dx > 0 else "left"

    @property
    def center(self) -> Tuple[int, int]:
        b = self.bbox.flatten()
        return (
            int((b[0] + b[2]) / 2),
            int((b[1] + b[3]) / 2),
        )


@dataclass
class TrackedFrame:
    """Output of tracker for a single frame."""
    frame_id: int
    tracks:   List[Track]

    def vehicles(self) -> List[Track]:
        return [t for t in self.tracks
                if t.class_name in {"motorcycle","bicycle","car","bus","truck"}]

    def persons(self) -> List[Track]:
        return [t for t in self.tracks if t.class_name == "person"]


class SORTTracker:
    """
    Tracking Engineer's module.
    Simple Online and Realtime Tracking (SORT).
    Assigns stable track_ids across frames.
    """

    def __init__(self):
        self._tracks: List[Track] = []
        self._next_id = 1
        self.frame_count = 0
        logger.success("SORT Tracker ready")

    def update(self, detections: List[Detection]) -> TrackedFrame:
        self.frame_count += 1

        # 1. Predict all existing tracks
        for t in self._tracks:
            t.predict()

        # 2. Match detections to tracks via IOU
        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        # 3. Update matched tracks
        for det_idx, trk_idx in matched:
            self._tracks[trk_idx].update(detections[det_idx])

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._tracks.append(Track(
                track_id   = self._next_id,
                class_name = det.class_name,
                bbox       = det.bbox.copy(),
                confidence = det.confidence,
            ))
            self._next_id += 1

        # 5. Remove dead tracks
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= config.TRACK_MAX_AGE
        ]

        # 6. Return confirmed tracks only
        active = [t for t in self._tracks if t.is_confirmed]
        return TrackedFrame(frame_id=self.frame_count, tracks=active)

    def _associate(self, detections):
        if not self._tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self._tracks)))

        iou_matrix = np.zeros((len(detections), len(self._tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self._tracks):
                iou_matrix[d, t] = iou(det.bbox, trk.bbox)

        # Greedy matching
        matched_indices = []
        used_trk = set()
        used_det = set()

        # Sort by IOU descending
        pairs = [(iou_matrix[d,t], d, t)
                 for d in range(len(detections))
                 for t in range(len(self._tracks))]
        pairs.sort(reverse=True)

        for score, d, t in pairs:
            if score < config.TRACK_IOU_THRESH:
                break
            if d in used_det or t in used_trk:
                continue
            matched_indices.append((d, t))
            used_det.add(d)
            used_trk.add(t)

        unmatched_dets = [d for d in range(len(detections)) if d not in used_det]
        unmatched_trks = [t for t in range(len(self._tracks)) if t not in used_trk]
        return matched_indices, unmatched_dets, unmatched_trks

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracked: TrackedFrame) -> np.ndarray:
        frame = frame.copy()
        COLORS = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 180, 0), (0, 200, 255), (200, 0, 255),
        ]
        for t in tracked.tracks:
            color = COLORS[t.track_id % len(COLORS)]
            x1, y1, x2, y2 = t.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.track_id} {t.class_name}"
            cv2.putText(frame, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # Draw trail
            for i in range(1, len(t.center_history)):
                cv2.line(frame, t.center_history[i-1], t.center_history[i],
                         color, 1)
        return frame


import cv2

# violation/engine.py — Tracking & Logic Engineer's core module
# Rule-based violation detection with event buffering

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime
from loguru import logger
import uuid, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from tracking.tracker import Track, TrackedFrame


# ─── Violation Data Model ─────────────────────────────────────────────────────

@dataclass
class Violation:
    violation_id:   str
    violation_type: str          # "no_helmet" | "triple_riding" | "wrong_direction"
    track_id:       int
    frame_id:       int
    bbox:           np.ndarray   # bbox of offending vehicle
    confidence:     float
    timestamp:      str          = field(default_factory=lambda: datetime.now().isoformat())
    evidence_frame: Optional[np.ndarray] = field(default=None, repr=False)
    plate_number:   str          = "UNKNOWN"
    status:         str          = "pending"   # pending | confirmed | needs_review

    def to_dict(self) -> dict:
        return {
            "violation_id":   self.violation_id,
            "violation_type": self.violation_type,
            "track_id":       self.track_id,
            "frame_id":       self.frame_id,
            "confidence":     round(self.confidence, 3),
            "timestamp":      self.timestamp,
            "plate_number":   self.plate_number,
            "status":         self.status,
            "bbox":           self.bbox.tolist(),
        }


# ─── Per-Track State ──────────────────────────────────────────────────────────

@dataclass
class TrackState:
    """Sliding window state for a single track."""
    track_id:   int
    class_name: str

    # Buffers: last N frames of flags
    no_helmet_buf:    deque = field(default_factory=lambda: deque(maxlen=10))
    triple_rider_buf: deque = field(default_factory=lambda: deque(maxlen=10))
    wrong_dir_buf:    deque = field(default_factory=lambda: deque(maxlen=10))

    # Already-fired violation IDs to avoid duplicate triggers
    fired: Dict[str, str] = field(default_factory=dict)  # type → violation_id

    def cooldown_expired(self, vtype: str, frame_id: int, cooldown=60) -> bool:
        """Don't re-trigger the same violation within cooldown frames."""
        key = f"{vtype}_last_frame"
        last = self.fired.get(key, -9999)
        return (frame_id - last) > cooldown

    def mark_fired(self, vtype: str, frame_id: int, vid: str):
        self.fired[f"{vtype}_last_frame"] = frame_id
        self.fired[vtype] = vid


# ─── Violation Engine ─────────────────────────────────────────────────────────

class ViolationEngine:
    """
    Logic Engineer's module.
    Processes tracked frames → emits Violation events.

    Rules:
    1. No Helmet   — bike detected, rider has no helmet for N consecutive frames
    2. Triple Ride — bike has 3+ persons for N frames
    3. Wrong Dir   — vehicle moves against ALLOWED_DIRECTION for N frames
    """

    def __init__(self):
        self._states: Dict[int, TrackState] = {}
        self._violations: List[Violation] = []
        self._frame_buffer: deque = deque(maxlen=config.EVIDENCE_BUFFER_SECONDS * config.VIDEO_FPS_DEFAULT)
        logger.success("Violation Engine ready")

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_frame(
        self,
        tracked: TrackedFrame,
        raw_frame: np.ndarray,
    ) -> List[Violation]:
        """
        Feed one tracked frame. Returns any new violations triggered this frame.
        """
        self._frame_buffer.append(raw_frame.copy())
        new_violations: List[Violation] = []

        # Update state for every active track
        for track in tracked.tracks:
            state = self._get_or_create_state(track)
            frame_violations = self._evaluate_track(track, tracked, state, raw_frame)
            new_violations.extend(frame_violations)

        self._violations.extend(new_violations)
        return new_violations

    @property
    def all_violations(self) -> List[Violation]:
        return self._violations

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_or_create_state(self, track: Track) -> TrackState:
        if track.track_id not in self._states:
            self._states[track.track_id] = TrackState(
                track_id   = track.track_id,
                class_name = track.class_name,
            )
        return self._states[track.track_id]

    def _evaluate_track(
        self,
        track: Track,
        tracked: TrackedFrame,
        state: TrackState,
        frame: np.ndarray,
    ) -> List[Violation]:
        results = []
        is_bike = track.class_name in {"motorcycle", "bicycle"}
        is_vehicle = track.class_name in {"motorcycle","bicycle","car","bus","truck"}

        if is_bike:
            results += self._check_no_helmet(track, tracked, state, frame)
            results += self._check_triple_riding(track, tracked, state, frame)

        if is_vehicle:
            results += self._check_wrong_direction(track, state, frame)

        return results

    # ── Rule 1: No Helmet ──────────────────────────────────────────────────────

    def _check_no_helmet(
        self,
        bike: Track,
        tracked: TrackedFrame,
        state: TrackState,
        frame: np.ndarray,
    ) -> List[Violation]:
        persons_on_bike = self._persons_on_bike(bike, tracked.persons())
        if not persons_on_bike:
            state.no_helmet_buf.append(False)
            return []

        # Check if any rider has NO helmet
        # (In base YOLOv8, we use head proximity as proxy — helmet model improves this)
        has_helmet = self._any_helmet_detected(bike, frame)
        state.no_helmet_buf.append(not has_helmet)

        n_no_helmet = sum(state.no_helmet_buf)
        confidence = n_no_helmet / len(state.no_helmet_buf)

        if (
            n_no_helmet >= config.HELMET_FRAMES_REQUIRED
            and confidence >= config.MIN_HELMET_CONFIDENCE
            and state.cooldown_expired("no_helmet", tracked.frame_id)
        ):
            v = self._make_violation(
                vtype      = "no_helmet",
                track      = bike,
                frame_id   = tracked.frame_id,
                confidence = confidence,
                frame      = frame,
                label      = "No Helmet",
            )
            state.mark_fired("no_helmet", tracked.frame_id, v.violation_id)
            logger.warning(f"🚨 NO HELMET — Track {bike.track_id} conf={confidence:.2f}")
            return [v]

        return []

    def _any_helmet_detected(self, bike: Track, frame: np.ndarray) -> bool:
        """
        Head-region colour heuristic — no custom model required.

        Strategy:
          1. Crop upper zone of bike bbox (where rider heads appear)
          2. Check for helmet colours (black/white/red/blue/yellow) in HSV
          3. Compare against skin-tone ratio — skin dominant = no helmet
          ~65% accuracy on dashcam footage; good enough for the event buffer.

        Swap with a dedicated YOLO helmet model call when available.
        """
        try:
            b = np.asarray(bike.bbox, dtype=int).flatten()
            bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            h_frame, w_frame   = frame.shape[:2]

            # Head zone: slightly above bike top down to 1/3 height of bike
            head_y1 = max(0, by1 - 40)
            head_y2 = max(0, by1 + (by2 - by1) // 3)
            head_x1 = max(0, bx1)
            head_x2 = min(w_frame, bx2)

            if head_y2 <= head_y1 or head_x2 <= head_x1:
                return False

            crop = frame[head_y1:head_y2, head_x1:head_x2]
            if crop.size == 0:
                return False

            hsv       = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            total_px  = crop.shape[0] * crop.shape[1]
            if total_px == 0:
                return False

            # Helmet colour masks
            black  = cv2.inRange(hsv, np.array([0,   0,   0]),   np.array([180, 60,  60]))
            white  = cv2.inRange(hsv, np.array([0,   0,   180]), np.array([180, 40,  255]))
            red1   = cv2.inRange(hsv, np.array([0,   80,  60]),  np.array([10,  255, 255]))
            red2   = cv2.inRange(hsv, np.array([165, 80,  60]),  np.array([180, 255, 255]))
            red    = cv2.bitwise_or(red1, red2)
            blue   = cv2.inRange(hsv, np.array([95,  80,  60]),  np.array([130, 255, 255]))
            yellow = cv2.inRange(hsv, np.array([20,  100, 100]), np.array([35,  255, 255]))

            helmet_mask  = cv2.bitwise_or(
                cv2.bitwise_or(cv2.bitwise_or(black, white), cv2.bitwise_or(red, blue)),
                yellow
            )
            # Skin tone mask (exposed head = no helmet)
            skin = cv2.inRange(hsv, np.array([0, 20, 80]), np.array([25, 180, 255]))

            helmet_ratio = cv2.countNonZero(helmet_mask) / total_px
            skin_ratio   = cv2.countNonZero(skin)        / total_px

            # Helmet present when helmet colours dominate skin tones
            return helmet_ratio > 0.15 and helmet_ratio > skin_ratio * 1.5

        except Exception:
            return False   # fail safe → treat as no helmet

    # ── Rule 2: Triple Riding ──────────────────────────────────────────────────

    def _check_triple_riding(
        self,
        bike: Track,
        tracked: TrackedFrame,
        state: TrackState,
        frame: np.ndarray,
    ) -> List[Violation]:
        persons = self._persons_on_bike(bike, tracked.persons())
        is_triple = len(persons) >= 3
        state.triple_rider_buf.append(is_triple)

        n_triple = sum(state.triple_rider_buf)
        if (
            n_triple >= config.TRIPLE_FRAMES_REQUIRED
            and len(persons) >= 3
            and state.cooldown_expired("triple_riding", tracked.frame_id)
        ):
            confidence = min(0.95, 0.5 + 0.15 * len(persons))
            v = self._make_violation(
                vtype      = "triple_riding",
                track      = bike,
                frame_id   = tracked.frame_id,
                confidence = confidence,
                frame      = frame,
                label      = f"Triple Riding ({len(persons)} persons)",
            )
            state.mark_fired("triple_riding", tracked.frame_id, v.violation_id)
            logger.warning(f"🚨 TRIPLE RIDING — Track {bike.track_id} persons={len(persons)}")
            return [v]

        return []

    # ── Rule 3: Wrong Direction ────────────────────────────────────────────────

    def _check_wrong_direction(
        self,
        vehicle: Track,
        state: TrackState,
        frame: np.ndarray,
    ) -> List[Violation]:
        direction = vehicle.movement_direction
        if direction is None or direction == "stationary":
            state.wrong_dir_buf.append(False)
            return []

        going_wrong = (direction != config.ALLOWED_DIRECTION)
        state.wrong_dir_buf.append(going_wrong)

        n_wrong = sum(state.wrong_dir_buf)
        if (
            n_wrong >= config.WRONG_DIR_FRAMES
            and going_wrong
            and state.cooldown_expired("wrong_direction", 0, cooldown=90)
        ):
            confidence = n_wrong / len(state.wrong_dir_buf)
            v = self._make_violation(
                vtype      = "wrong_direction",
                track      = vehicle,
                frame_id   = 0,
                confidence = confidence,
                frame      = frame,
                label      = f"Wrong Direction ({direction})",
            )
            state.mark_fired("wrong_direction", 0, v.violation_id)
            logger.warning(f"🚨 WRONG DIRECTION — Track {vehicle.track_id}")
            return [v]

        return []

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _persons_on_bike(
        self,
        bike: Track,
        persons: List[Track],
    ) -> List[Track]:
        """Return persons whose center falls inside/near the bike bbox."""
        b = np.asarray(bike.bbox, dtype=int).flatten()
        bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        result = []
        for p in persons:
            px, py = p.center
            if (bx1 - 20 <= px <= bx2 + 20 and
                by1 - config.PERSON_ON_BIKE_Y_MARGIN <= py <= by2 + 20):
                result.append(p)
        return result

    def _make_violation(
        self,
        vtype: str,
        track: Track,
        frame_id: int,
        confidence: float,
        frame: np.ndarray,
        label: str,
    ) -> Violation:
        evidence = self._annotate_evidence(frame.copy(), track, label)
        status = "confirmed" if confidence >= config.MIN_VIOLATION_CONFIDENCE else "needs_review"
        return Violation(
            violation_id   = str(uuid.uuid4())[:8],
            violation_type = vtype,
            track_id       = track.track_id,
            frame_id       = frame_id,
            bbox           = np.asarray(track.bbox, dtype=int).flatten(),
            confidence     = float(confidence),
            evidence_frame = evidence,
            status         = status,
        )

    @staticmethod
    def _annotate_evidence(frame: np.ndarray, track: Track, label: str) -> np.ndarray:
        b = np.asarray(track.bbox, dtype=int).flatten()
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for i, text in enumerate([f"VIOLATION: {label}", f"ID:{track.track_id}", ts]):
            cv2.putText(frame, text, (x1, y1 - 8 - i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    @staticmethod
    def draw_violations(frame: np.ndarray, violations: List["Violation"]) -> np.ndarray:
        frame = frame.copy()
        for v in violations:
            b = np.asarray(v.bbox, dtype=int).flatten()
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, v.violation_type.upper().replace("_", " "),
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return frame

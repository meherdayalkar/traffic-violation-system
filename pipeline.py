# pipeline.py — System Lead's integration module (YOUR file)
# Orchestrates: Detection → Tracking → Violation → ANPR → Challan
# This is the single thread that ties everything together.

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple, List
from dataclasses import dataclass
from loguru import logger
import time, sys, os

sys.path.insert(0, os.path.dirname(__file__))
import config
from detection.detector import VehicleDetector, Detection
from tracking.tracker   import SORTTracker, TrackedFrame
from violation.engine   import ViolationEngine, Violation
from anpr.plate_reader  import ANPRReader
from backend.challan    import ChallanGenerator, Challan


@dataclass
class PipelineFrame:
    """Everything about one processed frame — passed to UI."""
    frame_id:      int
    frame:         np.ndarray           # Annotated frame ready for display
    raw_frame:     np.ndarray           # Original
    detections:    List[Detection]
    tracked:       TrackedFrame
    new_violations: List[Violation]
    new_challans:   List[Challan]
    fps:           float = 0.0


class Pipeline:
    """
    System Lead's integration class.
    Wires every module into a single callable pipeline.

    Usage:
        pipeline = Pipeline()
        for pf in pipeline.process_video("video.mp4"):
            display(pf.frame)
    """

    def __init__(self, enable_anpr: bool = True):
        logger.info("Initialising pipeline...")
        self.detector  = VehicleDetector()
        self.tracker   = SORTTracker()
        self.engine    = ViolationEngine()
        self.challan   = ChallanGenerator()
        self.anpr      = ANPRReader() if enable_anpr else None
        self._enable_anpr = enable_anpr
        logger.success("Pipeline ready ✓")

    # ── Main entry points ──────────────────────────────────────────────────────

    def process_video(
        self,
        source,           # Path to video OR 0 for webcam
        max_frames: Optional[int] = None,
        skip_frames: int = 1,        # Process every Nth frame (1=all, 2=half, etc.)
    ) -> Generator[PipelineFrame, None, None]:
        """
        Generator: yields PipelineFrame for each processed frame.
        Caller controls display / streaming.
        """
        cap = cv2.VideoCapture(str(source) if source != 0 else 0)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_FPS_DEFAULT
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {source} | FPS={fps_video:.1f} | Frames={total}")

        frame_idx   = 0
        processed   = 0
        t_prev      = time.time()

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break
            if frame_idx % skip_frames != 0:
                continue

            pf = self.process_frame(raw_frame, frame_id=frame_idx)

            # FPS calc
            t_now = time.time()
            pf.fps = 1.0 / (t_now - t_prev + 1e-9)
            t_prev = t_now
            processed += 1

            yield pf

       

    def process_frame(self, raw_frame: np.ndarray, frame_id: int = 0) -> PipelineFrame:
        """Process a single numpy frame. Returns PipelineFrame."""
        frame = raw_frame.copy()

        # ── Stage 1: Detect ───────────────────────────────────────────────────
        detections = self.detector.detect(frame)

        # ── Stage 2: Track ────────────────────────────────────────────────────
        tracked = self.tracker.update(detections)

        # ── Stage 3: Violation logic ──────────────────────────────────────────
        new_violations = self.engine.process_frame(tracked, raw_frame)

        # ── Stage 4: ANPR (only on violations) ───────────────────────────────
        if self._enable_anpr and new_violations and self.anpr:
            for v in new_violations:
                result = self.anpr.read_from_bbox(raw_frame, v.bbox)
                if result.is_valid:
                    v.plate_number = result.text
                    logger.info(f"  Plate: {result.text} (conf={result.confidence:.2f})")
                else:
                    v.plate_number = "UNCLEAR"

        # ── Stage 5: Challan generation ───────────────────────────────────────
        new_challans = self.challan.generate_batch(new_violations)

        # ── Stage 6: Annotate display frame ──────────────────────────────────
        frame = self._annotate(frame, detections, tracked, new_violations)

        return PipelineFrame(
            frame_id       = frame_id,
            frame          = frame,
            raw_frame      = raw_frame,
            detections     = detections,
            tracked        = tracked,
            new_violations = new_violations,
            new_challans   = new_challans,
        )

    # ── Annotation ─────────────────────────────────────────────────────────────

    def _annotate(self, frame, detections, tracked, violations) -> np.ndarray:
        """Draw tracks + violations + HUD on frame."""
        # Draw tracks (colored boxes with IDs)
        TRACK_COLORS = {
            "motorcycle": (0, 140, 255),
            "bicycle":    (0, 140, 255),
            "person":     (255, 200, 0),
            "car":        (180, 180, 180),
            "bus":        (180, 180, 180),
            "truck":      (180, 180, 180),
        }
        for track in tracked.tracks:
            color = TRACK_COLORS.get(track.class_name, (200, 200, 200))
            x1, y1, x2, y2 = int(track.bbox[0]), int(track.bbox[1]), int(track.bbox[2]), int(track.bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.track_id} {track.class_name}"
            cv2.putText(frame, label, (x1, max(y1-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            # Motion trail
            for i in range(1, len(track.center_history)):
                cv2.line(frame,
                         track.center_history[i-1],
                         track.center_history[i],
                         color, 1)

        # Draw violations (red boxes)
        for v in violations:
            x1, y1, x2, y2 = int(v.bbox[0]), int(v.bbox[1]), int(v.bbox[2]), int(v.bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = v.violation_type.replace("_", " ").upper()
            cv2.putText(frame, label, (x1, max(y1-10, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            plate = v.plate_number
            cv2.putText(frame, plate, (x1, y2+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD overlay
        h, w = frame.shape[:2]
        hud = [
            f"Vehicles: {len(tracked.vehicles())}",
            f"Persons:  {len(tracked.persons())}",
            f"Violations: {len(self.engine.all_violations)}",
            f"Challans:   {len(self.challan.all_challans)}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 22 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        return frame

    # ── Helpers ────────────────────────────────────────────────────────────────

    def export_session(self) -> Path:
        return self.challan.export_session_json()

    @property
    def all_challans(self):
        return self.challan.all_challans

    @property
    def all_violations(self):
        return self.engine.all_violations

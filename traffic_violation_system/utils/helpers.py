# utils/helpers.py — Shared utility functions across all modules

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def resize_frame(frame: np.ndarray, width: int = 1280) -> np.ndarray:
    """Resize frame to given width while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    vehicle_count: int,
    violation_count: int,
    challan_count: int,
) -> np.ndarray:
    """Draw a heads-up display overlay on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent background strip
    cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    items = [
        f"FPS: {fps:.1f}",
        f"Vehicles: {vehicle_count}",
        f"Violations: {violation_count}",
        f"Challans: {challan_count}",
        datetime.now().strftime("%H:%M:%S"),
    ]
    x = 10
    for item in items:
        cv2.putText(frame, item, (x, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        x += len(item) * 10 + 20

    return frame


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IOU matrix between two sets of bboxes.
    bboxes_a: (N, 4), bboxes_b: (M, 4) — format [x1,y1,x2,y2]
    Returns: (N, M) IOU matrix
    """
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    inter_x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    inter_y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    inter_x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    inter_y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


def crop_bbox(
    frame: np.ndarray,
    bbox: np.ndarray,
    pad: int = 0,
) -> Optional[np.ndarray]:
    """Crop a region from a frame using a bbox, with optional padding."""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_clip(
    frames: list,
    path: Path,
    fps: float = 25.0,
) -> bool:
    """Save a list of numpy frames as an MP4 clip."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return True


def put_text_with_background(
    frame: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int,int,int] = (255, 255, 255),
    bg_color: Tuple[int,int,int] = (0, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw text with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x-2, y-th-4), (x+tw+2, y+baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    return frame


def get_video_info(path) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(str(path))
    info = {
        "fps":    cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["frames"] / max(info["fps"], 1)
    cap.release()
    return info

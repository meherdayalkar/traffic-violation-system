# config.py — Central configuration for the entire pipeline
# Edit these values to tune the system for your environment

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

CLIPS_DIR    = OUTPUT_DIR / "clips"
CHALLANS_DIR = OUTPUT_DIR / "challans"
FRAMES_DIR   = OUTPUT_DIR / "frames"

# Ensure output directories exist (needed for cloud deployment)
for _dir in [DATA_DIR, OUTPUT_DIR, CLIPS_DIR, CHALLANS_DIR, FRAMES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"          # yolov8n=fast, yolov8s=better accuracy
YOLO_CONFIDENCE = 0.45                  # Minimum detection confidence
YOLO_IOU        = 0.45                  # IOU threshold for NMS
DEVICE          = "cpu"                 # "cpu" for cloud deployment; change to "cuda" locally if GPU available

# YOLO class IDs from COCO (used before custom model)
COCO_CLASSES = {
    "person":     0,
    "bicycle":    1,
    "motorbike":  3,
    "car":        2,
    "bus":        5,
    "truck":      7,
}

# ─── Tracking ─────────────────────────────────────────────────────────────────
TRACK_MAX_AGE      = 10    # Frames before a lost track is dropped
TRACK_MIN_HITS     = 3     # Frames before a track is confirmed
TRACK_IOU_THRESH   = 0.3   # IOU threshold for track association

# ─── Violation Logic ──────────────────────────────────────────────────────────
HELMET_FRAMES_REQUIRED  = 5    # Consecutive no-helmet frames to trigger violation
TRIPLE_FRAMES_REQUIRED  = 3    # Frames of 3+ riders to trigger
WRONG_DIR_FRAMES        = 5    # Frames of wrong direction to trigger

PERSON_ON_BIKE_Y_MARGIN = 80   # px: how far below bike bbox top a person can be

# Confidence thresholds for violations
MIN_VIOLATION_CONFIDENCE  = 0.60
MIN_HELMET_CONFIDENCE     = 0.50

# ─── ANPR ─────────────────────────────────────────────────────────────────────
PLATE_CONFIDENCE_THRESHOLD = 0.4     # Min OCR confidence to accept plate text
PLATE_MIN_CHARS            = 4       # Reject plates shorter than this
PLATE_MAX_CHARS            = 12      # Reject plates longer than this

# ─── Evidence Capture ─────────────────────────────────────────────────────────
EVIDENCE_BUFFER_SECONDS = 3      # Seconds of video to save around violation
VIDEO_FPS_DEFAULT       = 25     # Fallback FPS if video metadata unavailable

# ─── Backend API ──────────────────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000

# ─── Direction Detection ──────────────────────────────────────────────────────
# Define allowed movement direction per region
# "right" = vehicles should move left→right, "left" = right→left, etc.
ALLOWED_DIRECTION = "right"   # Change based on your test video
DIRECTION_LINE_Y  = None      # Set to pixel Y value to use horizontal reference line

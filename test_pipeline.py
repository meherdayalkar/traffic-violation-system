# test_pipeline.py — Run this FIRST to verify your setup before launching UI
# Usage: python test_pipeline.py
# Usage with video: python test_pipeline.py --video path/to/video.mp4
# Usage with webcam: python test_pipeline.py --webcam

import sys
import os
import argparse
import cv2

# ── Arg parsing ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Traffic Violation Pipeline Test")
parser.add_argument("--video",  type=str, default=None, help="Path to video file")
parser.add_argument("--webcam", action="store_true",    help="Use webcam (index 0)")
parser.add_argument("--frames", type=int, default=200,  help="Max frames to process")
parser.add_argument("--skip",   type=int, default=2,    help="Process every Nth frame")
parser.add_argument("--no-anpr", action="store_true",   help="Disable ANPR (faster)")
parser.add_argument("--no-display", action="store_true",help="Headless mode (no window)")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("  Traffic Violation System — Pipeline Test")
print("=" * 60)

# ── Step 1: Import check ──────────────────────────────────────────────────────
print("\n[1/4] Checking imports...")
try:
    import numpy as np
    from loguru import logger
    import config
    print("  ✓ numpy, loguru, config")
except ImportError as e:
    print(f"  ✗ {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("  ✓ ultralytics (YOLOv8)")
except ImportError:
    print("  ✗ ultralytics — run: pip install ultralytics")
    sys.exit(1)

if not args.no_anpr:
    try:
        import easyocr
        print("  ✓ easyocr")
    except ImportError:
        print("  ⚠ easyocr not found — ANPR disabled. Install: pip install easyocr")
        args.no_anpr = True

# ── Step 2: Load pipeline ─────────────────────────────────────────────────────
print("\n[2/4] Loading pipeline modules...")
try:
    from pipeline import Pipeline
    pipeline = Pipeline(enable_anpr=not args.no_anpr)
    print("  ✓ Pipeline loaded successfully")
except Exception as e:
    print(f"  ✗ Pipeline failed to load: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Step 3: Determine source ──────────────────────────────────────────────────
print("\n[3/4] Setting up video source...")
if args.webcam:
    source = 0
    print("  Using webcam (index 0)")
elif args.video:
    if not os.path.exists(args.video):
        print(f"  ✗ Video not found: {args.video}")
        sys.exit(1)
    source = args.video
    from utils.helpers import get_video_info
    info = get_video_info(source)
    print(f"  Video: {args.video}")
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.1f}fps | {info['frames']} frames | {info['duration_sec']:.1f}s")
else:
    # Synthetic test: generate solid-color frames
    print("  No source specified — using synthetic test frames (10 black frames)")
    source = None

# ── Step 4: Run pipeline ──────────────────────────────────────────────────────
print("\n[4/4] Running pipeline...\n")

if source is not None:
    # Real video / webcam
    for i, pf in enumerate(pipeline.process_video(
        source,
        max_frames = args.frames,
        skip_frames = args.skip,
    )):
        if not args.no_display:
            cv2.imshow("Traffic Violation System — Press Q to quit", pf.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if i % 20 == 0:
            print(f"  Frame {pf.frame_id:4d} | FPS {pf.fps:5.1f} | "
                  f"Detections: {len(pf.detections):2d} | "
                  f"Tracks: {len(pf.tracked.tracks):2d} | "
                  f"Violations: {len(pipeline.all_violations):2d}")

    if not args.no_display:
        cv2.destroyAllWindows()

else:
    # Synthetic frames test
    for i in range(10):
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pf = pipeline.process_frame(fake_frame, frame_id=i+1)
        print(f"  Synthetic frame {i+1}: detections={len(pf.detections)} tracks={len(pf.tracked.tracks)}")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"  Total violations : {len(pipeline.all_violations)}")
print(f"  Total challans   : {len(pipeline.all_challans)}")

if pipeline.all_violations:
    print("\n  Violations detected:")
    for v in pipeline.all_violations:
        print(f"    [{v.status.upper()}] {v.violation_type} | "
              f"Track #{v.track_id} | Plate: {v.plate_number} | "
              f"Conf: {v.confidence:.2f}")

if pipeline.all_challans:
    print("\n  Challans generated:")
    for c in pipeline.all_challans:
        print(f"    {c.challan_id} | {c.violation_desc} | "
              f"Plate: {c.vehicle_number} | ₹{c.fine_amount:,}")
    session_file = pipeline.export_session()
    print(f"\n  Session exported → {session_file}")

print("\n✅ Pipeline test complete. If no errors above, you're good to launch the UI.")
print("   Run:  streamlit run app/main.py\n")

#!/usr/bin/env python
"""Quick import test to find errors."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("[1] Testing basic imports...")
try:
    import config
    print("  ✓ config")
except Exception as e:
    print(f"  ✗ config: {e}")
    sys.exit(1)

print("[2] Testing detection module...")
try:
    from detection.detector import VehicleDetector, Detection
    print("  ✓ detection.detector")
except Exception as e:
    print(f"  ✗ detection.detector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[3] Testing tracking module...")
try:
    from tracking.tracker import SORTTracker
    print("  ✓ tracking.tracker")
except Exception as e:
    print(f"  ✗ tracking.tracker: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[4] Testing violation module...")
try:
    from violation.engine import ViolationEngine
    print("  ✓ violation.engine")
except Exception as e:
    print(f"  ✗ violation.engine: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[5] Testing ANPR module...")
try:
    from anpr.plate_reader import ANPRReader
    print("  ✓ anpr.plate_reader")
except Exception as e:
    print(f"  ✗ anpr.plate_reader: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[6] Testing backend module...")
try:
    from backend.challan import ChallanGenerator
    print("  ✓ backend.challan")
except Exception as e:
    print(f"  ✗ backend.challan: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[7] Testing pipeline module...")
try:
    from pipeline import Pipeline
    print("  ✓ pipeline")
except Exception as e:
    print(f"  ✗ pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports successful!")

# backend/challan.py — Backend Engineer's challan generation module
# Converts Violation events into structured challan records + saves evidence

import json
import cv2
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from violation.engine import Violation


FINE_AMOUNTS = {
    "no_helmet":       1000,   # INR
    "triple_riding":   2000,
    "wrong_direction": 1500,
    "overspeed":       2000,
}

VIOLATION_LABELS = {
    "no_helmet":       "Riding without helmet",
    "triple_riding":   "Triple riding on two-wheeler",
    "wrong_direction": "Driving in wrong direction",
}


@dataclass
class Challan:
    challan_id:     str
    vehicle_number: str
    violation_type: str
    violation_desc: str
    fine_amount:    int
    timestamp:      str
    frame_id:       int
    track_id:       int
    confidence:     float
    status:         str
    evidence_path:  Optional[str] = None
    location:       str = "Nagpur Traffic Zone"   # set from GPS in real system
    officer_id:     str = "AUTO-SYSTEM-v1"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_display(self) -> dict:
        """Human-readable version for the UI."""
        return {
            "Challan ID":    self.challan_id,
            "Plate":         self.vehicle_number,
            "Violation":     self.violation_desc,
            "Fine (₹)":      f"₹{self.fine_amount:,}",
            "Time":          self.timestamp,
            "Confidence":    f"{self.confidence*100:.0f}%",
            "Status":        self.status.upper(),
        }


class ChallanGenerator:
    """
    Backend Engineer's module.
    Takes confirmed Violation objects → produces Challan records
    + saves evidence images + writes JSON logs.
    """

    def __init__(self):
        config.CHALLANS_DIR.mkdir(parents=True, exist_ok=True)
        config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        self._challans: List[Challan] = []
        self._counter = 1
        logger.success("Challan Generator ready")

    def generate(self, violation: Violation) -> Optional[Challan]:
        """Convert a Violation → Challan. Returns None if below confidence threshold."""
        if violation.confidence < config.MIN_VIOLATION_CONFIDENCE:
            logger.info(f"Skipping low-confidence violation {violation.violation_id} ({violation.confidence:.2f})")
            return None

        challan_id = f"NGP{datetime.now().strftime('%Y%m%d')}{self._counter:04d}"
        self._counter += 1

        evidence_path = None
        if violation.evidence_frame is not None:
            evidence_path = self._save_evidence(violation, challan_id)

        challan = Challan(
            challan_id     = challan_id,
            vehicle_number = violation.plate_number,
            violation_type = violation.violation_type,
            violation_desc = VIOLATION_LABELS.get(violation.violation_type, violation.violation_type),
            fine_amount    = FINE_AMOUNTS.get(violation.violation_type, 1000),
            timestamp      = violation.timestamp,
            frame_id       = violation.frame_id,
            track_id       = violation.track_id,
            confidence     = violation.confidence,
            status         = violation.status,
            evidence_path  = evidence_path,
        )

        self._challans.append(challan)
        self._save_challan_json(challan)
        logger.success(f"📋 Challan {challan_id} — {challan.violation_desc} — {violation.plate_number}")
        return challan

    def generate_batch(self, violations: List[Violation]) -> List[Challan]:
        """Process a list of violations, return generated challans."""
        challans = []
        for v in violations:
            c = self.generate(v)
            if c:
                challans.append(c)
        return challans

    @property
    def all_challans(self) -> List[Challan]:
        return self._challans

    def export_session_json(self, path: Optional[Path] = None) -> Path:
        """Dump all challans from this session to a single JSON file."""
        if path is None:
            path = config.CHALLANS_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            "session_time": datetime.now().isoformat(),
            "total_challans": len(self._challans),
            "challans": [c.to_dict() for c in self._challans],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Session exported → {path}")
        return path

    def _save_evidence(self, violation: Violation, challan_id: str) -> str:
        path = config.FRAMES_DIR / f"{challan_id}_{violation.violation_type}.jpg"
        cv2.imwrite(str(path), violation.evidence_frame)
        return str(path)

    def _save_challan_json(self, challan: Challan):
        path = config.CHALLANS_DIR / f"{challan.challan_id}.json"
        with open(path, "w") as f:
            json.dump(challan.to_dict(), f, indent=2)

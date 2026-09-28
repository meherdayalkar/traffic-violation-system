# anpr/plate_reader.py — Backend Engineer's ANPR module
# License plate detection via YOLO crop + EasyOCR

import cv2
import numpy as np
import easyocr
import re
from typing import Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


@dataclass
class PlateResult:
    text:       str
    confidence: float
    bbox:       Optional[np.ndarray] = None   # plate bbox within the vehicle crop
    raw_texts:  list = None

    @property
    def is_valid(self) -> bool:
        clean = re.sub(r'[^A-Z0-9]', '', self.text.upper())
        return (
            self.confidence >= config.PLATE_CONFIDENCE_THRESHOLD
            and config.PLATE_MIN_CHARS <= len(clean) <= config.PLATE_MAX_CHARS
        )


class ANPRReader:
    """
    Backend Engineer's ANPR module.
    Crops vehicle region → runs EasyOCR → validates plate text.
    """

    # Indian plate patterns (common formats)
    PLATE_PATTERNS = [
        r'[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}',   # MH12AB1234
        r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}',      # DL10AB1234
        r'\d{2}[A-Z]{1,2}\d{4}',            # partial
    ]

    def __init__(self):
        logger.info("Loading EasyOCR (first run downloads models ~200MB)...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            logger.success("ANPR Reader ready (GPU)")
        except Exception:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.success("ANPR Reader ready (CPU)")

    def read_plate(self, vehicle_frame: np.ndarray) -> PlateResult:
        """
        Given a cropped vehicle image, extract the plate text.
        Returns PlateResult (check .is_valid before trusting).
        """
        if vehicle_frame is None or vehicle_frame.size == 0:
            return PlateResult(text="UNKNOWN", confidence=0.0)

        # Preprocess: resize, enhance contrast
        processed = self._preprocess(vehicle_frame)

        # Try full crop first
        results = self.reader.readtext(processed)

        if not results:
            return PlateResult(text="UNCLEAR", confidence=0.0, raw_texts=[])

        # Score each candidate
        best = self._pick_best(results)
        return best

    def read_from_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        expand: float = 0.15,
    ) -> PlateResult:
        """
        Crop vehicle from full frame using bbox, then read plate.
        expand: fractional expansion of bbox for context.
        """
        h, w = frame.shape[:2]
        b = np.asarray(bbox, dtype=int).flatten()
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        dw = int((x2 - x1) * expand)
        dh = int((y2 - y1) * expand)
        x1 = int(max(0, x1 - dw))
        y1 = int(max(0, y1 - dh))
        x2 = int(min(w, x2 + dw))
        y2 = int(min(h, y2 + dh))
        crop = frame[y1:y2, x1:x2]

        # Focus on lower half of vehicle (where plates usually are)
        lower_half = crop[crop.shape[0]//2:, :]
        result = self.read_plate(lower_half)
        return result

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize + denoise + sharpen for better OCR."""
        # Upscale small crops
        h, w = img.shape[:2]
        if w < 200:
            scale = 200 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to remove noise while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Back to BGR for EasyOCR
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _pick_best(self, results) -> PlateResult:
        """
        From EasyOCR results [(bbox, text, conf)],
        pick the best candidate matching plate pattern.
        """
        raw_texts = [(text, conf) for (_, text, conf) in results]
        candidates = []

        for (_, text, conf) in results:
            cleaned = self._clean_plate(text)
            score = conf

            # Boost score if matches Indian plate pattern
            for pattern in self.PLATE_PATTERNS:
                if re.search(pattern, cleaned.upper()):
                    score += 0.25
                    break

            candidates.append((cleaned, score))

        if not candidates:
            return PlateResult(text="UNCLEAR", confidence=0.0, raw_texts=raw_texts)

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_text, best_conf = candidates[0]

        return PlateResult(
            text       = best_text.upper() if best_text else "UNCLEAR",
            confidence = min(best_conf, 1.0),
            raw_texts  = raw_texts,
        )

    @staticmethod
    def _clean_plate(text: str) -> str:
        """Remove spaces, special chars. Map common OCR errors."""
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9]', '', text)
        # Common OCR substitutions
        text = text.replace('O', '0').replace('I', '1').replace('S', '5')
        return text

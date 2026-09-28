# 🚦 Traffic Violation Detection System
**Prototype — Built for 1-Day Sprint**  
Stack: YOLOv8 + SORT Tracker + EasyOCR + FastAPI + Streamlit

---

## 👥 Team Roles → File Map

| Role | File(s) |
|------|---------|
| 👤 System Lead (YOU) | `pipeline.py`, `config.py` |
| 👤 CV Engineer | `detection/detector.py` |
| 👤 Tracking & Logic Engineer | `tracking/tracker.py`, `violation/engine.py` |
| 👤 Backend Engineer | `anpr/plate_reader.py`, `backend/challan.py`, `backend/server.py` |
| 👤 UI Engineer | `app/main.py` |

---

## 🚀 Quick Start (Windows, NVIDIA GPU)

### Step 1 — Setup (run ONCE)
```bat
setup.bat
```
This installs PyTorch CUDA, all dependencies, downloads YOLOv8n model.

### Step 2 — Test pipeline (no UI needed)
```bat
python test_pipeline.py --no-display
```
With your own video:
```bat
python test_pipeline.py --video C:\path\to\traffic.mp4
```

### Step 3 — Launch full UI
```bat
run_ui.bat
```
Or manually:
```bat
streamlit run app/main.py
```
Opens at: **http://localhost:8501**

### Step 4 — (Optional) Start API server
```bat
run_api.bat
```
API docs at: **http://localhost:8000/docs**

---

## 📁 Folder Structure

```
traffic_violation_system/
│
├── config.py               ← All tunable settings (edit this first)
├── pipeline.py             ← System Lead: integration orchestrator
├── test_pipeline.py        ← Validate setup before UI
│
├── detection/
│   └── detector.py         ← CV Engineer: YOLOv8 detection
│
├── tracking/
│   └── tracker.py          ← Tracking Engineer: SORT tracker
│
├── violation/
│   └── engine.py           ← Logic Engineer: rule-based violations + event buffer
│
├── anpr/
│   └── plate_reader.py     ← Backend: EasyOCR plate reading
│
├── backend/
│   ├── challan.py          ← Backend: challan generation + JSON export
│   └── server.py           ← Backend: FastAPI REST endpoints
│
├── app/
│   └── main.py             ← UI Engineer: Streamlit interface
│
├── utils/
│   └── helpers.py          ← Shared utilities
│
├── data/sample/            ← Put test videos here
├── output/
│   ├── clips/              ← Saved violation video clips
│   ├── challans/           ← JSON challan records
│   └── frames/             ← Evidence screenshots
│
├── requirements.txt
├── setup.bat               ← Windows one-click setup
├── run_ui.bat              ← Launch Streamlit UI
└── run_api.bat             ← Launch FastAPI server
```

---

## 🎯 Violations Detected

| Violation | Rule | Confidence |
|-----------|------|-----------|
| 🪖 No Helmet | Rider on bike, no helmet for 5+ frames | Medium |
| 👨‍👩‍👦 Triple Riding | 3+ persons on bike for 3+ frames | Medium |
| ⬅️ Wrong Direction | Vehicle moving against `ALLOWED_DIRECTION` for 5+ frames | Medium |

---

## ⚙️ Key Config Values (`config.py`)

```python
YOLO_MODEL      = "yolov8n.pt"    # Switch to yolov8s.pt for better accuracy
DEVICE          = "cuda"          # "cpu" if GPU fails
ALLOWED_DIRECTION = "right"       # Change to "left" based on your test video

HELMET_FRAMES_REQUIRED = 5        # Increase to reduce false positives
TRIPLE_FRAMES_REQUIRED = 3
MIN_VIOLATION_CONFIDENCE = 0.60   # Below this → "needs_review", not challan
```

---

## 🔌 Pipeline Data Flow

```
Video Frame
    │
    ▼
VehicleDetector.detect(frame)
    │  → List[Detection]  (bbox, class, confidence)
    ▼
SORTTracker.update(detections)
    │  → TrackedFrame  (tracks with stable IDs)
    ▼
ViolationEngine.process_frame(tracked, raw_frame)
    │  → List[Violation]  (type, track_id, evidence_frame)
    ▼
ANPRReader.read_from_bbox(frame, bbox)
    │  → PlateResult  (text, confidence)
    ▼
ChallanGenerator.generate(violation)
    │  → Challan  (JSON record + evidence image saved)
    ▼
Streamlit UI / FastAPI
```

---

## 🧩 Integration Points (System Lead Checklist)

- [ ] `config.py` — Set `ALLOWED_DIRECTION` to match your test video
- [ ] `config.py` — Set `DEVICE = "cpu"` if CUDA issues
- [ ] `test_pipeline.py` — Passes with your video before showing to team
- [ ] `violation/engine.py` → `_any_helmet_detected()` — Plug in custom helmet model here when ready
- [ ] `config.py` — Tune `HELMET_FRAMES_REQUIRED` if too many false positives

---

## 🛠️ Troubleshooting

**CUDA out of memory:**
```python
# config.py
DEVICE = "cpu"
# or
YOLO_MODEL = "yolov8n.pt"  # use nano, not large
```

**EasyOCR slow on first run:**  
Downloads ~200MB models. Normal. Subsequent runs use cache.

**No motorcycles detected:**  
YOLOv8 COCO classes: `motorcycle` = class 3, `bicycle` = class 1.  
If testing with dashcam footage, increase `YOLO_CONFIDENCE` threshold.

**filterpy import error:**
```bat
pip install filterpy==1.4.5
```

**Streamlit port in use:**
```bat
streamlit run app/main.py --server.port 8502
```

---

## 📊 Output Format

### Challan JSON
```json
{
  "challan_id": "NGP202401010001",
  "vehicle_number": "MH31AB1234",
  "violation_type": "no_helmet",
  "violation_desc": "Riding without helmet",
  "fine_amount": 1000,
  "timestamp": "2024-01-01T10:30:00",
  "confidence": 0.82,
  "status": "confirmed",
  "location": "Nagpur Traffic Zone",
  "evidence_path": "output/frames/NGP202401010001_no_helmet.jpg"
}
```

---

## ⚠️ Prototype Limitations

| Limitation | Reality |
|-----------|---------|
| Helmet detection | Proxy only — needs custom trained model |
| Plate accuracy | ~60-70% on clear plates |
| Overspeed | Not implemented — avoid faking it |
| Legal validity | None — this is a prototype |

---

## 🔮 Next Steps (Post-Prototype)

1. Train custom helmet detection model (Roboflow dataset available)
2. Integrate proper plate detection YOLO model (not just OCR)
3. Replace SORT with ByteTrack for better multi-object tracking
4. Add GPS metadata to challan records
5. Connect to RTO database for vehicle owner lookup

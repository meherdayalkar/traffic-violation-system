# backend/server.py — REST API for the pipeline
# Run: uvicorn backend.server:app --reload

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil, tempfile, os, sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from backend.challan import ChallanGenerator

app = FastAPI(
    title="Traffic Violation System API",
    description="Prototype — Nagpur Traffic",
    version="1.0.0",
)

_challan_gen = ChallanGenerator()


# ─── Models ───────────────────────────────────────────────────────────────────

class ViolationIn(BaseModel):
    violation_type: str
    track_id:       int
    frame_id:       int
    confidence:     float
    plate_number:   str = "UNKNOWN"
    status:         str = "pending"
    bbox:           List[int]


class PipelineStatus(BaseModel):
    status:          str
    total_challans:  int
    session_id:      str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/challans", summary="List all challans this session")
def list_challans():
    return [c.to_dict() for c in _challan_gen.all_challans]


@app.get("/challans/{challan_id}")
def get_challan(challan_id: str):
    for c in _challan_gen.all_challans:
        if c.challan_id == challan_id:
            return c.to_dict()
    raise HTTPException(status_code=404, detail="Challan not found")


@app.get("/challans/{challan_id}/evidence")
def get_evidence(challan_id: str):
    for c in _challan_gen.all_challans:
        if c.challan_id == challan_id and c.evidence_path:
            return FileResponse(c.evidence_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Evidence not found")


@app.post("/violations/register", summary="Register a violation and generate challan")
def register_violation(v: ViolationIn):
    """Called by the pipeline when a new violation is detected."""
    import numpy as np
    from violation.engine import Violation
    import uuid
    from datetime import datetime

    viol = Violation(
        violation_id   = str(uuid.uuid4())[:8],
        violation_type = v.violation_type,
        track_id       = v.track_id,
        frame_id       = v.frame_id,
        bbox           = np.array(v.bbox),
        confidence     = v.confidence,
        plate_number   = v.plate_number,
        status         = v.status,
        timestamp      = datetime.now().isoformat(),
    )
    challan = _challan_gen.generate(viol)
    if challan:
        return {"success": True, "challan_id": challan.challan_id}
    return {"success": False, "reason": "Below confidence threshold"}


@app.get("/export")
def export_session():
    path = _challan_gen.export_session_json()
    return FileResponse(str(path), media_type="application/json",
                        filename=path.name)


@app.get("/stats")
def stats():
    challans = _challan_gen.all_challans
    by_type = {}
    for c in challans:
        by_type[c.violation_type] = by_type.get(c.violation_type, 0) + 1
    return {
        "total": len(challans),
        "by_type": by_type,
        "total_fine": sum(c.fine_amount for c in challans),
    }

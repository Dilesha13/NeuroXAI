from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models.patient import Patient
from app.db.models.eeg_record import EEGRecord
from app.db.models.inference_result import InferenceResult
from app.db.models.report import Report

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


CONFIDENCE_BUCKETS = [
    (0.9, 1.01, "90-100%"),
    (0.8, 0.9, "80-90%"),
    (0.7, 0.8, "70-80%"),
    (0.6, 0.7, "60-70%"),
    (0.0, 0.6, "<60%"),
]


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _format_duration(duration_seconds: float | None) -> str:
    if not duration_seconds:
        return "—"
    minutes = max(1, round(duration_seconds / 60))
    return f"{minutes} min"


def _decision_to_status(decision: str | None) -> tuple[str, str]:
    normalized = (decision or "").lower()
    if "seizure" in normalized:
        return "Seizure Detected", "Abnormal"
    return "Normal", "Normal"


@router.get("/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    patients = db.query(Patient).all()
    records = db.query(EEGRecord).all()
    inferences = db.query(InferenceResult).all()
    reports = db.query(Report).all()

    records_by_id = {record.id: record for record in records}
    patient_by_id = {patient.id: patient for patient in patients}
    latest_report_by_inference_id = {report.inference_result_id: report for report in reports}

    total_analyses = len(inferences)
    seizure_detections = sum(1 for inf in inferences if "seizure" in (inf.overall_decision or "").lower())
    normal_recordings = total_analyses - seizure_detections
    active_patients = len({record.patient_id for record in records})

    ordered_months: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for inf in sorted(inferences, key=lambda item: _to_datetime(item.created_at) or datetime.min):
        created_at = _to_datetime(inf.created_at)
        if created_at is None:
            continue
        key = created_at.strftime("%Y-%m")
        label = created_at.strftime("%b")
        if key not in ordered_months:
            ordered_months[key] = {"month": label, "detections": 0, "normal": 0}
        if "seizure" in (inf.overall_decision or "").lower():
            ordered_months[key]["detections"] += 1
        else:
            ordered_months[key]["normal"] += 1
    seizure_trends = list(ordered_months.values())[-6:]

    confidence_distribution = []
    for low, high, label in CONFIDENCE_BUCKETS:
        count = sum(1 for inf in inferences if low <= float(inf.max_probability or 0.0) < high)
        confidence_distribution.append({"range": label, "count": count})

    recent = []
    sorted_inferences = sorted(inferences, key=lambda item: _to_datetime(item.created_at) or datetime.min, reverse=True)
    for inf in sorted_inferences[:8]:
        record = records_by_id.get(inf.eeg_record_id)
        if record is None:
            continue
        patient = patient_by_id.get(record.patient_id)
        status, result = _decision_to_status(inf.overall_decision)
        created_at = _to_datetime(inf.created_at) or _to_datetime(record.created_at)
        report = latest_report_by_inference_id.get(inf.id)
        recent.append(
            {
                "record_id": record.id,
                "inference_id": inf.id,
                "patient_id": record.patient_id,
                "patient_code": patient.patient_code if patient else f"Patient {record.patient_id}",
                "patient_name": patient.display_name if patient and patient.display_name else None,
                "date": created_at.isoformat() if created_at else None,
                "duration": _format_duration(record.duration_seconds),
                "status": status,
                "result": result,
                "confidence": round(float(inf.max_probability or 0.0) * 100, 1),
                "report_id": report.id if report else None,
                "download_url": f"/api/v1/reports/download/{report.id}" if report else None,
            }
        )

    return {
        "stats": {
            "total_eeg_analyses": total_analyses,
            "seizure_detections": seizure_detections,
            "normal_recordings": normal_recordings,
            "active_patients": active_patients,
        },
        "seizure_trends": seizure_trends,
        "confidence_distribution": confidence_distribution,
        "distribution": [
            {"name": "Seizure", "value": seizure_detections},
            {"name": "Normal", "value": normal_recordings},
        ],
        "recent_analyses": recent,
    }

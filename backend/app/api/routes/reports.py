from __future__ import annotations

from pathlib import Path
import uuid
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models.inference_result import InferenceResult
from app.db.models.eeg_record import EEGRecord
from app.db.models.patient import Patient
from app.db.models.explanation import Explanation
from app.db.models.report import Report
from app.services.report_service import ReportService
from app.api.deps import get_current_user
from app.db.models.user import User

router = APIRouter(prefix="/reports", tags=["Reports"])


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_inference_summary(inf: InferenceResult, exp: Explanation | None = None) -> dict:
    overall_decision = getattr(inf, "overall_decision", "Unknown")
    confidence_level = str(getattr(inf, "confidence_level", "") or "").lower()

    if overall_decision == "Seizure Detected" and "low" in confidence_level and "very high" not in confidence_level:
        overall_decision = "Possible Seizure Activity — Review Needed"

    summary = {
        "overall_decision": overall_decision,
        "num_windows": getattr(inf, "num_windows", 0),
        "num_seizure_windows": getattr(inf, "num_positive_windows", 0),
        "max_probability": _safe_float(getattr(inf, "max_probability", 0.0)),
        "mean_probability": _safe_float(getattr(inf, "mean_probability", 0.0)),
        "threshold": _safe_float(getattr(inf, "threshold", 0.5), 0.5),
        "window_sec": 10.0,
        "step_sec": 10.0,
        "probabilities": [],
        "top_channels": [],
        "seizure_ranges": [],
    }

    if exp and exp.top_channels_json:
        try:
            payload = json.loads(exp.top_channels_json)
            summary["top_channels"] = payload.get("top_channels", [])
            summary["seizure_ranges"] = payload.get("seizure_ranges", [])
            summary["probabilities"] = payload.get("probability_timeline", [])
        except Exception as e:
            print(f"[WARN] Failed to parse explanation JSON: {e}")

    return summary


@router.post("/{inference_id}/generate")
def generate_report(
    inference_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inf = db.query(InferenceResult).filter(InferenceResult.id == inference_id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Inference not found")

    rec = db.query(EEGRecord).filter(EEGRecord.id == inf.eeg_record_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="EEG record not found")

    patient = db.query(Patient).filter(Patient.id == rec.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    exp = db.query(Explanation).filter(Explanation.inference_result_id == inference_id).first()

    inference_summary = _build_inference_summary(inf, exp)
    explain_summary = exp.summary_text if exp and exp.summary_text else "No explanation summary available."

    duration_minutes = None
    if hasattr(rec, "duration_seconds") and getattr(rec, "duration_seconds") is not None:
        duration_minutes = _safe_float(getattr(rec, "duration_seconds")) / 60.0
    elif hasattr(rec, "duration_minutes") and getattr(rec, "duration_minutes") is not None:
        duration_minutes = _safe_float(getattr(rec, "duration_minutes"))

    recording_date = None
    if hasattr(rec, "recorded_at") and getattr(rec, "recorded_at") is not None:
        recording_date = str(getattr(rec, "recorded_at"))
    elif hasattr(rec, "created_at") and getattr(rec, "created_at") is not None:
        recording_date = str(getattr(rec, "created_at"))

    report_data = ReportService.build_report_data(
        patient_code=patient.patient_code,
        original_filename=rec.original_filename,
        inference_summary=inference_summary,
        explain_summary=explain_summary,
        recording_date=recording_date,
        duration_minutes=duration_minutes,
    )

    preferred_format = (getattr(current_user, "export_format", "PDF") or "PDF").upper()

    if preferred_format == "CSV":
        report_name = f"{uuid.uuid4().hex}_report.csv"
        output_path = ReportService.generate_csv(
            patient_code=patient.patient_code,
            original_filename=rec.original_filename,
            inference_summary=inference_summary,
            explain_summary=explain_summary,
            report_name=report_name,
            recording_date=recording_date,
            duration_minutes=duration_minutes,
        )
        report_type = "csv"

    elif preferred_format == "JSON":
        report_name = f"{uuid.uuid4().hex}_report.json"
        output_path = ReportService.generate_json(
            patient_code=patient.patient_code,
            original_filename=rec.original_filename,
            inference_summary=inference_summary,
            explain_summary=explain_summary,
            report_name=report_name,
            recording_date=recording_date,
            duration_minutes=duration_minutes,
        )
        report_type = "json"

    else:
        report_name = f"{uuid.uuid4().hex}_report.pdf"
        output_path = ReportService.generate_pdf(
            patient_code=patient.patient_code,
            original_filename=rec.original_filename,
            inference_summary=inference_summary,
            explain_summary=explain_summary,
            report_name=report_name,
            recording_date=recording_date,
            duration_minutes=duration_minutes,
        )
        report_type = "pdf"

    rep = Report(
        inference_result_id=inference_id,
        report_type=report_type,
        report_path=str(output_path),
    )
    db.add(rep)
    db.commit()
    db.refresh(rep)

    return {
        "report_id": rep.id,
        "report_path": rep.report_path,
        "download_url": f"/api/v1/reports/download/{rep.id}",
        "report_data": report_data,
    }


@router.get("/download/{report_id}")
def download_report(report_id: int, db: Session = Depends(get_db)):
    rep = db.query(Report).filter(Report.id == report_id).first()
    if not rep:
        raise HTTPException(status_code=404, detail="Report not found")

    p = Path(rep.report_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Report file missing")

    media_type = "application/octet-stream"
    if rep.report_type == "pdf":
        media_type = "application/pdf"
    elif rep.report_type == "csv":
        media_type = "text/csv"
    elif rep.report_type == "json":
        media_type = "application/json"

    return FileResponse(
        str(p),
        media_type=media_type,
        filename=p.name,
    )


@router.get("/{inference_id}")
def get_report_preview(inference_id: int, db: Session = Depends(get_db)):
    inf = db.query(InferenceResult).filter(InferenceResult.id == inference_id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Inference not found")

    rec = db.query(EEGRecord).filter(EEGRecord.id == inf.eeg_record_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="EEG record not found")

    patient = db.query(Patient).filter(Patient.id == rec.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    exp = db.query(Explanation).filter(Explanation.inference_result_id == inference_id).first()

    inference_summary = _build_inference_summary(inf, exp)
    explain_summary = exp.summary_text if exp and exp.summary_text else "No explanation summary available."

    duration_minutes = None
    if hasattr(rec, "duration_seconds") and getattr(rec, "duration_seconds") is not None:
        duration_minutes = _safe_float(getattr(rec, "duration_seconds")) / 60.0
    elif hasattr(rec, "duration_minutes") and getattr(rec, "duration_minutes") is not None:
        duration_minutes = _safe_float(getattr(rec, "duration_minutes"))

    recording_date = None
    if hasattr(rec, "recorded_at") and getattr(rec, "recorded_at") is not None:
        recording_date = str(getattr(rec, "recorded_at"))
    elif hasattr(rec, "created_at") and getattr(rec, "created_at") is not None:
        recording_date = str(getattr(rec, "created_at"))

    report_data = ReportService.build_report_data(
        patient_code=patient.patient_code,
        original_filename=rec.original_filename,
        inference_summary=inference_summary,
        explain_summary=explain_summary,
        recording_date=recording_date,
        duration_minutes=duration_minutes,
    )

    return {
        "inference_id": inference_id,
        "report_data": report_data,
    }
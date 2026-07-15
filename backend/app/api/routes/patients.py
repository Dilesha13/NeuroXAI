from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.patient import PatientCreate, PatientRead
from app.services.patient_service import PatientService
from app.db.models.eeg_record import EEGRecord
from app.db.models.inference_result import InferenceResult

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("", response_model=PatientRead)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    return PatientService.create_patient(db, payload)


@router.get("", response_model=list[PatientRead])
def list_patients(db: Session = Depends(get_db)):
    return PatientService.list_patients(db)


@router.get("/records")
def list_patient_records(db: Session = Depends(get_db)):
    patients = {p.id: p for p in PatientService.list_patients(db)}
    records = db.query(EEGRecord).order_by(EEGRecord.created_at.desc()).all()
    inferences = db.query(InferenceResult).all()

    latest_inference_by_record_id = {}
    for inf in inferences:
        existing = latest_inference_by_record_id.get(inf.eeg_record_id)
        if existing is None or (existing.created_at or datetime.min) < (inf.created_at or datetime.min):
            latest_inference_by_record_id[inf.eeg_record_id] = inf

    payload = []
    for r in records:
        p = patients.get(r.patient_id)
        inf = latest_inference_by_record_id.get(r.id)

        duration_minutes = round((r.duration_seconds or 0) / 60) if r.duration_seconds else None

        decision = (getattr(inf, "overall_decision", "") or "").lower()
        is_seizure = "seizure" in decision
        confidence = round(float(getattr(inf, "max_probability", 0.0) or 0.0) * 100, 1) if inf else None

        payload.append({
            "record_id": r.id,
            "patient_id": r.patient_id,
            "patient_code": p.patient_code if p and getattr(p, "patient_code", None) else f"P-{r.patient_id:03d}",
            "patient_name": p.display_name if p and getattr(p, "display_name", None) else f"Patient {r.patient_id}",
            "recording_date": (inf.created_at if inf and inf.created_at else r.created_at).isoformat()
            if ((inf and inf.created_at) or r.created_at)
            else None,
            "duration_minutes": duration_minutes,
            "duration_label": f"{duration_minutes} min" if duration_minutes else "—",
            "status": "Seizure Detected" if is_seizure else ("Completed" if inf else "Completed"),
            "confidence": confidence,
            "result": "Abnormal" if is_seizure else ("Normal" if inf else "Available"),
            "record_status": r.status,
            "report_id": None,
            "download_url": None,
            "inference_id": inf.id if inf else None,
            "filename": r.original_filename,
        })

    return payload


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = PatientService.get_patient(db, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient
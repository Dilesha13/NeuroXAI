from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.models.patient import Patient
from app.db.models.eeg_record import EEGRecord
from app.schemas.eeg_record import EEGRecordRead
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/eeg-records", tags=["EEG Records"])


def _validate_patient(db: Session, patient_id: int) -> Patient:
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def _validate_edf_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only EDF files are supported")


def _store_uploaded_file(file: UploadFile) -> Path:
    file_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    save_path = settings.uploads_dir / f"{file_id}_{safe_name}"

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    return save_path


def _create_eeg_record(db: Session, patient_id: int, file: UploadFile, save_path: Path) -> EEGRecord:
    record = EEGRecord(
        patient_id=patient_id,
        original_filename=file.filename,
        stored_path=str(save_path),
        status="uploaded",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.post("/upload", response_model=EEGRecordRead)
async def upload_eeg(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _validate_patient(db, patient_id)
    _validate_edf_file(file)

    save_path = _store_uploaded_file(file)
    record = _create_eeg_record(db, patient_id, file, save_path)
    return record


@router.post("/upload-and-analyze")
async def upload_and_analyze_eeg(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _validate_patient(db, patient_id)
    _validate_edf_file(file)

    save_path = _store_uploaded_file(file)
    record = _create_eeg_record(db, patient_id, file, save_path)

    service = InferenceService()

    try:
        result = service.run_for_record(db, record)
        return {
            "success": True,
            "message": "EDF uploaded and analysed successfully",
            "record": {
                "id": record.id,
                "patient_id": record.patient_id,
                "original_filename": record.original_filename,
                "stored_path": record.stored_path,
                "status": record.status,
                "duration_seconds": record.duration_seconds,
                "sampling_rate_original": record.sampling_rate_original,
            },
            "result": result,
        }
    except Exception as e:
        record.status = "failed"
        db.add(record)
        db.commit()

        raise HTTPException(
            status_code=500,
            detail=f"Upload succeeded but analysis failed: {str(e)}",
        )
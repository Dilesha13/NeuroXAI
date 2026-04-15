from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models.eeg_record import EEGRecord
from app.db.models.inference_result import InferenceResult
from app.db.models.explanation import Explanation
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/inference", tags=["Inference"])
service = InferenceService()


@router.post("/run/{record_id}")
def run_inference(record_id: int, db: Session = Depends(get_db)):
    record = db.query(EEGRecord).filter(EEGRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="EEG record not found")

    try:
        record.status = "processing"
        db.add(record)
        db.commit()
        db.refresh(record)

        result = service.run_for_record(db, record)
        return result

    except HTTPException:
        record.status = "failed"
        db.add(record)
        db.commit()
        raise
    except Exception as e:
        record.status = "failed"
        db.add(record)
        db.commit()

        # Keep backend log readable
        print(f"[ERROR] Inference failed for record_id={record_id}: {e}")

        raise HTTPException(
            status_code=500,
            detail="Inference failed while processing the EEG record."
        )


@router.get("/{inference_id}")
def get_inference(inference_id: int, db: Session = Depends(get_db)):
    inf = db.query(InferenceResult).filter(InferenceResult.id == inference_id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Inference result not found")

    exp = db.query(Explanation).filter(Explanation.inference_result_id == inference_id).first()

    return {
        "id": inf.id,
        "eeg_record_id": inf.eeg_record_id,
        "model_name": inf.model_name,
        "checkpoint_name": inf.checkpoint_name,
        "threshold": inf.threshold,
        "num_windows": inf.num_windows,
        "num_positive_windows": inf.num_positive_windows,
        "max_probability": inf.max_probability,
        "mean_probability": inf.mean_probability,
        "overall_decision": inf.overall_decision,
        "overall_summary": inf.overall_summary,
        "timeline_json_path": inf.timeline_json_path,
        "explanation": None if exp is None else {
            "top_channels_json": exp.top_channels_json,
            "saliency_path": exp.saliency_path,
            "temporal_attention_path": exp.temporal_attention_path,
            "gat_attention_path": exp.gat_attention_path,
            "summary_text": exp.summary_text,
        },
    }
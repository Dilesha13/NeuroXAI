from sqlalchemy.orm import Session
from app.db.models.patient import Patient
from app.schemas.patient import PatientCreate

class PatientService:
    @staticmethod
    def create_patient(db: Session, payload: PatientCreate) -> Patient:
        obj = Patient(**payload.model_dump())
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    @staticmethod
    def list_patients(db: Session):
        return db.query(Patient).order_by(Patient.id.desc()).all()

    @staticmethod
    def get_patient(db: Session, patient_id: int):
        return db.query(Patient).filter(Patient.id == patient_id).first()

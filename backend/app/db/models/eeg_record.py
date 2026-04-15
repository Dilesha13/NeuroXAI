from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.sql import func
from app.db.session import Base

class EEGRecord(Base):
    __tablename__ = 'eeg_records'

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    status = Column(String(32), nullable=False, default='uploaded')
    duration_seconds = Column(Float, nullable=True)
    sampling_rate_original = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

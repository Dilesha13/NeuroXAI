from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.sql import func
from app.db.session import Base

class InferenceResult(Base):
    __tablename__ = 'inference_results'

    id = Column(Integer, primary_key=True, index=True)
    eeg_record_id = Column(Integer, ForeignKey('eeg_records.id'), nullable=False, index=True)
    model_name = Column(String(64), nullable=False)
    checkpoint_name = Column(String(255), nullable=False)
    threshold = Column(Float, nullable=False)
    num_windows = Column(Integer, nullable=False)
    num_positive_windows = Column(Integer, nullable=False)
    max_probability = Column(Float, nullable=False)
    mean_probability = Column(Float, nullable=False)
    overall_decision = Column(String(64), nullable=False)
    overall_summary = Column(Text, nullable=True)
    timeline_json_path = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

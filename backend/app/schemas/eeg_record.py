from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class EEGRecordRead(BaseModel):
    id: int
    patient_id: int
    original_filename: str
    stored_path: str
    status: str
    duration_seconds: Optional[float] = None
    sampling_rate_original: Optional[float] = None
    created_at: datetime

    model_config = {'from_attributes': True}

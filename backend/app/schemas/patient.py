from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PatientCreate(BaseModel):
    patient_code: str
    display_name: Optional[str] = None
    sex: Optional[str] = None
    notes: Optional[str] = None

class PatientRead(BaseModel):
    id: int
    patient_code: str
    display_name: Optional[str] = None
    sex: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime

    model_config = {'from_attributes': True}

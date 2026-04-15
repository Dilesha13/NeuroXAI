from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ProfileRead(BaseModel):
    id: int
    full_name: str
    email: str
    hospital: Optional[str] = None
    specialization: Optional[str] = None
    profile_photo_url: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProfileUpdate(BaseModel):
    full_name: str
    email: str
    hospital: Optional[str] = None
    specialization: Optional[str] = None


class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


class PreferencesRead(BaseModel):
    email_alerts: bool
    seizure_detection_alerts: bool
    weekly_reports: bool
    system_updates: bool
    export_format: str
    data_retention: str

    model_config = {"from_attributes": True}


class PreferencesUpdate(BaseModel):
    email_alerts: bool
    seizure_detection_alerts: bool
    weekly_reports: bool
    system_updates: bool
    export_format: str
    data_retention: str


class MessageResponse(BaseModel):
    message: str
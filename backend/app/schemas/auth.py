from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class SignUpRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    hospital: Optional[str] = None
    specialization: Optional[str] = None


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class MessageResponse(BaseModel):
    message: str


class AuthUserRead(BaseModel):
    id: int
    full_name: str
    email: str
    hospital: Optional[str] = None
    specialization: Optional[str] = None
    profile_photo_url: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUserRead
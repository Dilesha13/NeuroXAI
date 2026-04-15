from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.auth import (
    AuthResponse,
    AuthUserRead,
    MessageResponse,
    SignInRequest,
    SignUpRequest,
    VerifyEmailRequest,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/signup", response_model=MessageResponse)
def sign_up(payload: SignUpRequest, db: Session = Depends(get_db)):
    return AuthService.sign_up(db, payload)


@router.post("/verify-email", response_model=MessageResponse)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    return AuthService.verify_email(db, payload)


@router.post("/signin", response_model=AuthResponse)
def sign_in(payload: SignInRequest, db: Session = Depends(get_db)):
    return AuthService.sign_in(db, payload)


@router.get("/me", response_model=AuthUserRead)
def get_me(current_user=Depends(get_current_user)):
    return AuthService.build_user_response(current_user)
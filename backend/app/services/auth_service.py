from datetime import datetime, timedelta, timezone
from pathlib import Path
import secrets

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.core.config import settings
from app.db.models.user import User
from app.schemas.auth import SignInRequest, SignUpRequest, VerifyEmailRequest
from app.services.email_service import EmailService


class AuthService:
    ALLOWED_EMAIL_DOMAINS = {"hospital.com", "nhs.uk", "gmail.com"}

    @staticmethod
    def sign_up(db: Session, payload: SignUpRequest):
        email = payload.email.lower().strip()
        domain = email.split("@")[-1]

        if domain not in AuthService.ALLOWED_EMAIL_DOMAINS:
            raise HTTPException(
                status_code=400,
                detail="Please use an authorized hospital email",
            )

        existing = db.query(User).filter(User.email == email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        verification_token = secrets.token_urlsafe(32)
        verification_expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.verification_token_expire_hours
        )

        user = User(
            full_name=payload.full_name,
            email=email,
            hospital=payload.hospital,
            specialization=payload.specialization,
            password_hash=hash_password(payload.password),
            is_verified=False,
            verification_token=verification_token,
            verification_expires_at=verification_expires_at,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        try:
            EmailService.send_verification_email(user.email, verification_token)
        except Exception as e:
            print("EMAIL SEND ERROR:", str(e))
            db.delete(user)
            db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to send verification email: {str(e)}",
            )

        return {
            "message": "Signup successful. Please verify your email before signing in."
        }

    @staticmethod
    def verify_email(db: Session, payload: VerifyEmailRequest):
        user = db.query(User).filter(User.verification_token == payload.token).first()

        if not user:
            raise HTTPException(status_code=400, detail="Invalid verification token")

        if user.is_verified:
            raise HTTPException(status_code=400, detail="Email already verified")

        if (
            user.verification_expires_at is None
            or user.verification_expires_at < datetime.now(timezone.utc)
        ):
            raise HTTPException(status_code=400, detail="Verification token has expired")

        user.is_verified = True
        user.verification_token = None
        user.verification_expires_at = None

        db.commit()

        return {"message": "Email verified successfully. You can now sign in."}

    @staticmethod
    def sign_in(db: Session, payload: SignInRequest):
        email = payload.email.lower().strip()

        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not user.is_active:
            raise HTTPException(status_code=403, detail="Inactive user")

        if not user.is_verified:
            raise HTTPException(
                status_code=403,
                detail="Please verify your email before signing in",
            )

        token = create_access_token(user.id)
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": AuthService.build_user_response(user),
        }

    @staticmethod
    def build_user_response(user: User):
        profile_photo_url = None

        if getattr(user, "profile_photo_path", None):
            filename = Path(user.profile_photo_path).name
            profile_photo_url = f"/static/profile_photos/{filename}"

        return {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "hospital": user.hospital,
            "specialization": user.specialization,
            "profile_photo_url": profile_photo_url,
            "is_active": user.is_active,
            "created_at": user.created_at,
        }
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.security import hash_password, verify_password
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.settings import (
    MessageResponse,
    PasswordUpdate,
    PreferencesRead,
    PreferencesUpdate,
    ProfileRead,
    ProfileUpdate,
)
from app.services.retention_service import RetentionService


router = APIRouter(prefix="/settings", tags=["Settings"])

def _profile_photo_url(current_user: User) -> str | None:
    if not getattr(current_user, "profile_photo_path", None):
        return None

    filename = Path(current_user.profile_photo_path).name
    return f"/static/profile_photos/{filename}"


@router.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "hospital": current_user.hospital,
        "specialization": current_user.specialization,
        "profile_photo_url": _profile_photo_url(current_user),
        "created_at": current_user.created_at,
    }


@router.put("/profile", response_model=ProfileRead)
def update_profile(
    payload: ProfileUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    existing = db.query(User).filter(User.email == payload.email, User.id != current_user.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already in use")

    current_user.full_name = payload.full_name
    current_user.email = payload.email
    current_user.hospital = payload.hospital
    current_user.specialization = payload.specialization

    db.commit()
    db.refresh(current_user)
    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "hospital": current_user.hospital,
        "specialization": current_user.specialization,
        "profile_photo_url": _profile_photo_url(current_user),
        "created_at": current_user.created_at,
    }

@router.post("/profile/photo")
def upload_profile_photo(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    photos_dir = Path("storage") / "profile_photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{ext}"
    out_path = photos_dir / filename

    with open(out_path, "wb") as f:
        f.write(file.file.read())

    current_user.profile_photo_path = str(out_path)
    db.commit()
    db.refresh(current_user)

    return {
        "message": "Profile photo uploaded successfully",
        "profile_photo_url": f"/static/profile_photos/{filename}",
    }

@router.get("/preferences", response_model=PreferencesRead)
def get_preferences(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/preferences", response_model=PreferencesRead)
def update_preferences(
    payload: PreferencesUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    current_user.email_alerts = payload.email_alerts
    current_user.seizure_detection_alerts = payload.seizure_detection_alerts
    current_user.weekly_reports = payload.weekly_reports
    current_user.system_updates = payload.system_updates
    current_user.export_format = payload.export_format
    current_user.data_retention = payload.data_retention

    db.commit()
    db.refresh(current_user)
    return current_user


@router.put("/password", response_model=MessageResponse)
def update_password(
    payload: PasswordUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="New password and confirm password do not match")

    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.password_hash = hash_password(payload.new_password)
    db.commit()

    return {"message": "Password updated successfully"}

@router.post("/run-retention-cleanup")
def run_retention_cleanup(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = RetentionService.cleanup_old_analysis_data(
        db=db,
        retention_value=current_user.data_retention,
    )
    return result


@router.delete("/account", response_model=MessageResponse)
def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted successfully"}
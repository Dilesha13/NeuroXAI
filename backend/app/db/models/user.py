from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    profile_photo_path = Column(String(512), nullable=True)
    full_name = Column(String(128), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hospital = Column(String(255), nullable=True)
    specialization = Column(String(255), nullable=True)

    password_hash = Column(String(255), nullable=False)

    email_alerts = Column(Boolean, nullable=False, default=True)
    seizure_detection_alerts = Column(Boolean, nullable=False, default=True)
    weekly_reports = Column(Boolean, nullable=False, default=False)
    system_updates = Column(Boolean, nullable=False, default=True)

    export_format = Column(String(32), nullable=False, default="PDF")
    data_retention = Column(String(32), nullable=False, default="3 months")

    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    is_verified = Column(Boolean, nullable=False, default=False)
    verification_token = Column(String(255), nullable=True)
    verification_expires_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
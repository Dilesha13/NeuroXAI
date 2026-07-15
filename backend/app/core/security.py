import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import settings


ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )
    return base64.b64encode(salt + derived).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    raw = base64.b64decode(password_hash.encode("utf-8"))
    salt = raw[:16]
    stored = raw[16:]
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )
    return hmac.compare_digest(derived, stored)


def create_access_token(user_id: int, expires_minutes: int | None = None) -> str:
    expire_minutes = expires_minutes or settings.access_token_expire_minutes
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)

    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "access",
    }

    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
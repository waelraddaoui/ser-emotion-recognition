from datetime import datetime, timedelta, timezone
import hashlib
import uuid
from jose import jwt, JWTError
from passlib.context import CryptContext
from .config import settings

ALGORITHM = "HS256"

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def _create_token(payload: dict, expires_delta: timedelta) -> str:
    to_encode = payload.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def create_access_token(user_id: int, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "jti": uuid.uuid4().hex,
    }
    expires = timedelta(minutes=settings.access_token_expire_minutes)
    return _create_token(payload, expires)


def create_refresh_token(user_id: int, role: str) -> tuple[str, str, datetime]:
    jti = uuid.uuid4().hex
    payload = {
        "sub": str(user_id),
        "role": role,
        "type": "refresh",
        "jti": jti,
    }
    expires = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    token = jwt.encode({**payload, "exp": expires}, settings.secret_key, algorithm=ALGORITHM)
    return token, jti, expires


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_token_payload(token: str) -> dict:
    try:
        return decode_token(token)
    except JWTError as exc:
        raise ValueError("Invalid token") from exc

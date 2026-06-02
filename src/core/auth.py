"""
auth.py — register / login / token helpers for ScholAI
"""

import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from src.models.database import fetch_one, execute
from src.common.logger   import get_logger

logger = get_logger(__name__)

#  config (override via env in production) 
JWT_SECRET    = "scholai-super-secret-change-in-prod"
JWT_ALGORITHM = "HS256"
JWT_EXP_HOURS = 72           # token valid for 3 days


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def _check_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))

def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub":   user_id,
        "email": email,
        "exp":   datetime.now(timezone.utc) + timedelta(hours=JWT_EXP_HOURS),
        "iat":   datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    """
    Returns the decoded payload dict or raises:
      jwt.ExpiredSignatureError  — token has expired
      jwt.InvalidTokenError      — tampered / malformed
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


#  register 
def register_user(name: str, email: str, password: str) -> dict:
    """
    Create a new user.
    Returns {"ok": True, "token": ..., "user": {...}}
         or {"ok": False, "error": "..."}
    """
    name     = name.strip()
    email    = email.strip().lower()
    password = password.strip()

    if not name:
        return {"ok": False, "error": "Name is required."}
    if not _is_valid_email(email):
        return {"ok": False, "error": "Invalid email address."}
    if len(password) < 6:
        return {"ok": False, "error": "Password must be at least 6 characters."}

    # duplicate check
    existing = fetch_one("SELECT id FROM users WHERE email = ?", (email,))
    if existing:
        return {"ok": False, "error": "An account with this email already exists."}

    try:
        password_hash = _hash_password(password)
        user_id = execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )
        token = create_token(user_id, email)
        logger.info("New user registered: %s (id=%d)", email, user_id)
        return {
            "ok":    True,
            "token": token,
            "user":  {"id": user_id, "name": name, "email": email}
        }
    except Exception as e:
        logger.error("Register error: %s", e)
        return {"ok": False, "error": "Registration failed. Please try again."}


#  login 
def login_user(email: str, password: str) -> dict:
    """
    Authenticate existing user.
    Returns {"ok": True, "token": ..., "user": {...}}
         or {"ok": False, "error": "..."}
    """
    email    = email.strip().lower()
    password = password.strip()

    if not email or not password:
        return {"ok": False, "error": "Email and password are required."}

    row = fetch_one(
        "SELECT id, name, email, password_hash FROM users WHERE email = ?",
        (email,)
    )
    if not row or not _check_password(password, row["password_hash"]):
        return {"ok": False, "error": "Invalid email or password."}

    token = create_token(row["id"], row["email"])
    logger.info("User logged in: %s (id=%d)", email, row["id"])
    return {
        "ok":    True,
        "token": token,
        "user":  {"id": row["id"], "name": row["name"], "email": row["email"]}
    }
"""
Sukoon - Authentication module.
Handles password hashing (passlib) and login/signup validation.
"""

from passlib.context import CryptContext

# passlib context for secure password hashing (bcrypt)
# Bcrypt accepts at most 72 bytes; we truncate to avoid ValueError
BCRYPT_MAX_BYTES = 72
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _truncate_for_bcrypt(password: str) -> str:
    """Truncate password to 72 bytes (UTF-8) so bcrypt accepts it."""
    encoded = password.encode("utf-8")
    if len(encoded) <= BCRYPT_MAX_BYTES:
        return password
    return encoded[:BCRYPT_MAX_BYTES].decode("utf-8", errors="ignore")


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt. Truncates to 72 bytes if longer."""
    return pwd_context.hash(_truncate_for_bcrypt(password))


def verify_password(plain_password: str, hashed: str) -> bool:
    """Verify a plain-text password against a hash. Truncates input to 72 bytes if longer."""
    try:
        return pwd_context.verify(_truncate_for_bcrypt(plain_password), hashed)
    except Exception:
        return False


def validate_username(username: str) -> tuple[bool, str]:
    """
    Validate username. Returns (is_valid, error_message).
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 32:
        return False, "Username must be at most 32 characters."
    if not username.replace("_", "").isalnum():
        return False, "Username can only contain letters, numbers, and underscores."
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength. Returns (is_valid, error_message).
    """
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if len(password) > 128:
        return False, "Password too long."
    return True, ""

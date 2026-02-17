import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet


def _derive_key(raw_key: str) -> bytes:
    digest = hashlib.sha256(raw_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def get_fernet(raw_key: Optional[str]) -> Fernet:
    material = raw_key or "dev-insecure-key"
    return Fernet(_derive_key(material))


def encrypt_value(value: str, raw_key: Optional[str]) -> str:
    fernet = get_fernet(raw_key)
    token = fernet.encrypt(value.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_value(token: str, raw_key: Optional[str]) -> str:
    fernet = get_fernet(raw_key)
    value = fernet.decrypt(token.encode("utf-8"))
    return value.decode("utf-8")

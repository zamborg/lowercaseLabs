import json
import os
import time

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm

APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"
APPLE_AUDIENCE = os.getenv("APPLE_BUNDLE_ID", "com.lowercaselabs.blackhole")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 90


async def verify_apple_token(identity_token: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(APPLE_KEYS_URL)
        keys_data = response.json()

    unverified_header = jwt.get_unverified_header(identity_token)
    kid = unverified_header.get("kid")

    matching_key = next((k for k in keys_data["keys"] if k["kid"] == kid), None)
    if not matching_key:
        raise ValueError("No matching Apple public key found")

    public_key = RSAAlgorithm.from_jwk(json.dumps(matching_key))
    payload = jwt.decode(
        identity_token,
        public_key,
        algorithms=["RS256"],
        audience=APPLE_AUDIENCE,
        issuer=APPLE_ISSUER,
    )
    return payload["sub"]


def create_session_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_DAYS * 86400,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_session_token(token: str) -> str:
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return payload["sub"]

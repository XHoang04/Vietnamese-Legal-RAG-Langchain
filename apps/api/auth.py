"""API Key authentication dependency."""

import os

from fastapi import Header, HTTPException, status


def get_valid_api_keys() -> set[str]:
    """Retrieve allowed API keys from environment variable API_KEYS."""
    raw = os.environ.get("API_KEYS", "dev-key-1")
    return {k.strip() for k in raw.split(",") if k.strip()}


async def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> str:
    """
    Validate the X-API-Key header.
    Raises 401 Unauthorized if missing or invalid.
    """
    valid_keys = get_valid_api_keys()
    if not x_api_key or x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Khóa API không hợp lệ hoặc bị thiếu trong tiêu đề X-API-Key",
            },
        )
    return x_api_key

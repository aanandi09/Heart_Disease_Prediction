"""
HTTP client for the CardioANN FastAPI backend.
Every function returns a tuple: (success: bool, data_or_error_message).
"""

import os
from typing import Any, Optional, Tuple

import httpx

API_BASE_URL = os.getenv("CARDIOANN_API_URL", "http://127.0.0.1:8000")
TIMEOUT_SECONDS = 20.0

BACKEND_DOWN_MESSAGE = (
    "Backend server is not reachable. Start it from the project root with: "
    "uvicorn backend.main:app --reload"
)


def _request(
    method: str,
    path: str,
    json: Optional[dict] = None,
    params: Optional[dict] = None,
) -> Tuple[bool, Any]:
    try:
        response = httpx.request(
            method,
            f"{API_BASE_URL}{path}",
            json=json,
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.RequestError:
        return False, BACKEND_DOWN_MESSAGE

    try:
        data = response.json()
    except ValueError:
        data = None

    if response.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        if isinstance(detail, list):
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        return False, detail or f"Request failed with status {response.status_code}."

    return True, data


def api_get(path: str, params: Optional[dict] = None) -> Tuple[bool, Any]:
    return _request("GET", path, params=params)


def api_post(path: str, payload: dict) -> Tuple[bool, Any]:
    return _request("POST", path, json=payload)


def api_put(path: str, payload: dict) -> Tuple[bool, Any]:
    return _request("PUT", path, json=payload)


def api_delete(path: str) -> Tuple[bool, Any]:
    return _request("DELETE", path)
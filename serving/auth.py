"""Authentication and authorization helpers for the serving API."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

from fastapi import HTTPException, Request, status

from serving.config import settings


@dataclass(frozen=True)
class Principal:
    api_key: str
    role: str
    subject: str
    authenticated: bool = True


class AuthManager:
    def __init__(self) -> None:
        self._api_key_roles: dict[str, str] = {}
        for key in settings.api_keys:
            self._api_key_roles[key] = "investigator"
        for key in settings.admin_api_keys:
            self._api_key_roles[key] = "admin"

        self.enabled = settings.auth_required
        self.demo_mode = (
            settings.allow_demo_auth
            and os.getenv("MULE_API_KEYS") is None
            and os.getenv("MULE_ADMIN_API_KEYS") is None
        )

    def principal_from_request(self, request: Request) -> Principal:
        token = self._extract_token(request)
        if not token:
            if self.enabled:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing API key. Provide X-API-Key or Bearer token.",
                )
            return Principal(api_key="", role="anonymous", subject="anonymous", authenticated=False)

        role = self._api_key_roles.get(token)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )

        return Principal(api_key=token, role=role, subject=f"{role}:{token[:6]}...")

    def _extract_token(self, request: Request) -> str | None:
        header_token = request.headers.get("X-API-Key", "").strip()
        if header_token:
            return header_token

        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            return auth_header.split(None, 1)[1].strip()
        return None

    def ensure_role(self, principal: Principal, allowed_roles: Iterable[str]) -> None:
        allowed = tuple(allowed_roles)
        if principal.role == "admin":
            return
        if principal.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this endpoint.",
            )

    def public_principal(self) -> Principal:
        if self.enabled:
            return Principal(api_key="", role="anonymous", subject="anonymous", authenticated=False)
        return Principal(api_key="", role="anonymous", subject="anonymous", authenticated=False)

    @staticmethod
    def request_id(request: Request) -> str:
        request_id = request.headers.get("X-Request-ID", "").strip()
        return request_id or str(uuid4())


auth_manager = AuthManager()

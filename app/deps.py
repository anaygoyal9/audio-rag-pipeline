from __future__ import annotations

import jwt
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .errors import AppError
from .security import decode_access_token
from .services import Services

bearer_scheme = HTTPBearer(auto_error=False, description="JWT from POST /api/v1/auth/login")
_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


def get_services(request: Request) -> Services:
    return request.app.state.services


async def _user_from_token(token: str | None, services: Services) -> dict:
    if not token:
        raise AppError(401, "not_authenticated", "Authentication required", headers=_WWW_AUTH)
    try:
        claims = decode_access_token(services.settings, token)
    except jwt.ExpiredSignatureError:
        raise AppError(401, "token_expired", "Token has expired", headers=_WWW_AUTH) from None
    except jwt.PyJWTError:
        raise AppError(401, "invalid_token", "Invalid token", headers=_WWW_AUTH) from None
    try:
        user_id = ObjectId(claims["sub"])
    except (InvalidId, TypeError):
        raise AppError(401, "invalid_token", "Invalid token", headers=_WWW_AUTH) from None
    user = await services.db.users.find_one({"_id": user_id})
    if not user or not user.get("is_active", True):
        raise AppError(401, "invalid_token", "Account not found or disabled", headers=_WWW_AUTH)
    return user


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    services: Services = Depends(get_services),
) -> dict:
    return await _user_from_token(creds.credentials if creds else None, services)


async def get_current_user_stream(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    services: Services = Depends(get_services),
) -> dict:
    """Like get_current_user, but also accepts ``?token=`` because a browser <audio> element
    cannot attach an Authorization header."""
    token = creds.credentials if creds else request.query_params.get("token")
    return await _user_from_token(token, services)


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise AppError(403, "forbidden", "Administrator access required")
    return user

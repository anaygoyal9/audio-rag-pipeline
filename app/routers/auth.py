from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from pymongo.errors import DuplicateKeyError

from ..deps import get_current_user, get_services
from ..errors import AppError
from ..schemas import TokenOut, UserCreate, UserLogin, UserOut, UserUpdate, serialize_user
from ..security import create_access_token, hash_password, verify_password
from ..services import Services
from ..utils import utcnow

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
async def register(body: UserCreate, s: Services = Depends(get_services)):
    now = utcnow()
    doc = {
        "email": body.email.lower(),
        "password_hash": await hash_password(body.password, s.settings.bcrypt_rounds),
        "display_name": body.display_name,
        "role": "user",  # admins are created out-of-band (scripts/create_admin.py)
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = await s.db.users.insert_one(doc)
    except DuplicateKeyError:
        raise AppError(409, "email_taken", "An account with this email already exists") from None
    doc["_id"] = result.inserted_id
    return serialize_user(doc)


@router.post("/login", response_model=TokenOut)
async def login(body: UserLogin, s: Services = Depends(get_services)):
    user = await s.db.users.find_one({"email": body.email.lower()})
    ok = await verify_password(body.password, user["password_hash"] if user else None, s.settings.bcrypt_rounds)
    if not user or not ok or not user.get("is_active", True):
        # Same message for unknown email and wrong password: no account enumeration.
        raise AppError(401, "invalid_credentials", "Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    await s.db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login_at": utcnow()}})
    token, ttl = create_access_token(s.settings, str(user["_id"]), user.get("role", "user"))
    return TokenOut(access_token=token, expires_in=ttl)


@router.get("/me", response_model=UserOut)
async def me(user: dict = Depends(get_current_user)):
    return serialize_user(user)


@router.patch("/me", response_model=UserOut)
async def update_me(body: UserUpdate, user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    changes: dict = {}
    if body.display_name is not None:
        changes["display_name"] = body.display_name
    if body.password is not None:
        changes["password_hash"] = await hash_password(body.password, s.settings.bcrypt_rounds)
    if not changes:
        raise AppError(422, "no_changes", "Provide display_name and/or password")
    changes["updated_at"] = utcnow()
    await s.db.users.update_one({"_id": user["_id"]}, {"$set": changes})
    user.update(changes)
    return serialize_user(user)


@router.delete("/me", status_code=204)
async def delete_me(user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    """Account deletion also removes the user's playlists and listening history."""
    await s.db.playlists.delete_many({"owner_id": user["_id"]})
    await s.db.listening_history.delete_many({"user_id": user["_id"]})
    await s.db.users.delete_one({"_id": user["_id"]})
    return Response(status_code=204)

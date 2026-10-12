"""Create (or promote) an administrator.

    python -m scripts.create_admin you@example.com
    (prompts for the password; nothing is read from or written to the shell history)
"""

from __future__ import annotations

import asyncio
import getpass
import sys

from app.config import get_settings
from app.db import ensure_indexes
from app.security import hash_password
from app.services import Services
from app.utils import utcnow


async def main(email: str) -> None:
    password = getpass.getpass("Password (min 8 chars): ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters")
    settings = get_settings()
    services = await Services.create(settings)
    await ensure_indexes(services.db)
    now = utcnow()
    await services.db.users.update_one(
        {"email": email.lower()},
        {
            "$set": {
                "role": "admin",
                "is_active": True,
                "password_hash": await hash_password(password, settings.bcrypt_rounds),
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now, "display_name": "Admin"},
        },
        upsert=True,
    )
    print(f"{email.lower()} is now an administrator")
    await services.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.create_admin <email>")
    asyncio.run(main(sys.argv[1]))

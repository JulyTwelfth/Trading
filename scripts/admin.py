import sys
import uuid
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta

from app.infra.dependencies import supabase


def generate(email: str | None = None) -> None:
    key = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + relativedelta(months=1)
    supabase.table("licenses").insert(
        {
            "key": key,
            "active": True,
            "user_email": email,
            "expires_at": expires_at.isoformat(),
        }
    ).execute()
    print(f"Generated: {key}")
    print(f"Expires:   {expires_at.strftime('%Y-%m-%d %H:%M UTC')}")


def revoke(key: str) -> None:
    supabase.table("licenses").update({"active": False}).eq("key", key).execute()
    print(f"Revoked: {key}")


def renew(key: str) -> None:
    response = supabase.table("licenses").select("expires_at").eq("key", key).execute()
    if not response.data:
        print(f"No license found: {key}")
        sys.exit(1)

    current_expires_at = datetime.fromisoformat(response.data[0]["expires_at"])
    now = datetime.now(timezone.utc)
    base = current_expires_at if current_expires_at > now else now
    new_expires_at = base + relativedelta(months=1)

    supabase.table("licenses").update({"expires_at": new_expires_at.isoformat()}).eq(
        "key", key
    ).execute()
    print(f"Renewed: {key}")
    print(f"Expires: {new_expires_at.strftime('%Y-%m-%d %H:%M UTC')}")


def list_keys() -> None:
    response = (
        supabase.table("licenses")
        .select("key, active, expires_at, user_email, created_at")
        .execute()
    )
    if not response.data:
        print("No license keys found.")
        return
    for row in response.data:
        status = "active" if row["active"] else "revoked"
        print(f"{row['key']}  {status}  email={row['user_email']}")
        print(f"  expires={row['expires_at']}  created={row['created_at']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/admin.py [generate|revoke|renew|list] [args]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        email = sys.argv[2] if len(sys.argv) > 2 else None
        generate(email)
    elif command == "revoke":
        if len(sys.argv) < 3:
            print("Usage: python scripts/admin.py revoke <key>")
            sys.exit(1)
        revoke(sys.argv[2])
    elif command == "renew":
        if len(sys.argv) < 3:
            print("Usage: python scripts/admin.py renew <key>")
            sys.exit(1)
        renew(sys.argv[2])
    elif command == "list":
        list_keys()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

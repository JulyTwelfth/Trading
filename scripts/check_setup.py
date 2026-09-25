"""Read-only validation for the local Supabase configuration.

This script does not place orders, register wallets, or print any secret.
"""

from urllib.parse import urlparse

REQUIRED_COLUMNS = {
    "licenses": "key",
    "sessions": "key",
    "wallets": "license_key",
    "blacklisted_markets": "license_key",
}


def main() -> int:
    try:
        from app.infra.config import settings
        from app.infra.dependencies import supabase
    except Exception as exc:
        print(f"FAIL configuration: {exc}")
        print("Fill Trading/.env from Trading/.env.example, then retry.")
        return 1

    host = urlparse(settings.supabase_url).hostname or "invalid URL"
    print(f"Supabase project: {host}")

    failed = False
    for table, column in REQUIRED_COLUMNS.items():
        try:
            supabase.table(table).select(column).limit(1).execute()
        except Exception as exc:
            failed = True
            print(f"FAIL table {table}: {exc}")
        else:
            print(f"OK   table {table}")

    if failed:
        print("Setup incomplete. Run supabase/schema.sql in the Supabase SQL Editor.")
        return 1

    print("Supabase connection and required tables are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

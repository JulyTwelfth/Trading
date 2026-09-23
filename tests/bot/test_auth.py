"""Bug 7 (POL-31): build_clob_client must try derive_api_key first, not
create_api_key, so we don't generate a 400 error log on every farm start when
the wallet already has an API key."""

from app.bot import auth as auth_mod
from app.bot.auth import build_clob_client

SENTINEL_DERIVED = "creds-from-derive"
SENTINEL_CREATED = "creds-from-create"


def install_fake_clob_client(monkeypatch, *, derive_succeeds: bool, create_succeeds: bool):
    """Replace ClobClient with a fake that records the call order. Returns the
    call-log list and the FakeClobClient instance after construction."""
    calls: list = []

    class FakeClobClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.creds = None

        def derive_api_key(self):
            calls.append("derive")
            if not derive_succeeds:
                raise RuntimeError("derive failed (no key exists yet)")
            return SENTINEL_DERIVED

        def create_api_key(self):
            calls.append("create")
            if not create_succeeds:
                raise RuntimeError("create failed")
            return SENTINEL_CREATED

        # The buggy SDK helper. If old code accidentally got re-introduced,
        # tests would call this and produce the wrong sentinel.
        def create_or_derive_api_key(self, nonce=None):
            try:
                r = self.create_api_key()
                if r:
                    return r
            except Exception:
                pass
            return self.derive_api_key()

        def set_api_creds(self, creds):
            calls.append(f"set_api_creds:{creds}")
            self.creds = creds

    monkeypatch.setattr(auth_mod, "ClobClient", FakeClobClient)
    return calls


def test_build_clob_client_derives_first_when_key_exists(monkeypatch):
    calls = install_fake_clob_client(monkeypatch, derive_succeeds=True, create_succeeds=True)

    client = build_clob_client("0xprivatekey", "0xproxy")

    # Derive must be the FIRST API call — that's the whole point of this fix.
    assert calls[0] == "derive"
    # Create must NOT be called when derive succeeded.
    assert "create" not in calls
    # The creds returned must be the derived ones.
    assert client.creds == SENTINEL_DERIVED


def test_build_clob_client_falls_back_to_create_when_derive_fails(monkeypatch):
    calls = install_fake_clob_client(monkeypatch, derive_succeeds=False, create_succeeds=True)

    client = build_clob_client("0xprivatekey", "0xproxy")

    # Order matters: derive first (fails), then create.
    assert calls[0] == "derive"
    assert calls[1] == "create"
    assert client.creds == SENTINEL_CREATED


def test_build_clob_client_propagates_when_both_fail(monkeypatch):
    install_fake_clob_client(monkeypatch, derive_succeeds=False, create_succeeds=False)

    import pytest

    with pytest.raises(RuntimeError, match="create failed"):
        build_clob_client("0xprivatekey", "0xproxy")


def test_build_clob_client_disables_resend(monkeypatch):
    # POL-62: retry_on_error must be False — the 30ms resend re-POSTs identical signed
    # orders (Duplicated 400s + double-sell risk); the held-share sweep is the retry.
    install_fake_clob_client(monkeypatch, derive_succeeds=True, create_succeeds=True)

    client = build_clob_client("0xprivatekey", "0xproxy")

    assert client.kwargs.get("retry_on_error") is False

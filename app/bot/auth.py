from py_clob_client_v2.client import ClobClient

from app.constants import CLOB_HOST, POLYGON_CHAIN_ID


def build_clob_client(
    private_key: str,
    proxy_address: str | None = None,
    signature_type: int = 2,
) -> ClobClient:
    client = ClobClient(
        host=CLOB_HOST,
        key=private_key,
        chain_id=POLYGON_CHAIN_ID,
        signature_type=signature_type,
        funder=proxy_address,
        retry_on_error=False,
    )
    try:
        creds = client.derive_api_key()
    except Exception:
        creds = client.create_api_key()
    client.set_api_creds(creds)
    return client

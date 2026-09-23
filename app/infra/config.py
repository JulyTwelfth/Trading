from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_key: str
    polygon_rpc_url: str = "https://polygon-rpc.com"
    force_legacy_execution: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

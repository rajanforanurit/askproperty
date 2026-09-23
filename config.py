import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DB_SERVER: str = os.getenv("DB_SERVER", "")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USERNAME: str = os.getenv("DB_USERNAME", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_DRIVER: str = os.getenv("DB_DRIVER", "pymssql")

    PRODUCT_TABLE: str = os.getenv("REBA_PRODUCT_TABLE", "Reba_Product")

    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))

    ACTIVE_STATUSES: list = os.getenv("ACTIVE_STATUSES", "Active,Stabilizing,Lease-Up").split(",")

    DEFAULT_NEARBY_COUNT: int = int(os.getenv("DEFAULT_NEARBY_COUNT", "5"))
    MAX_NEARBY_COUNT: int = int(os.getenv("MAX_NEARBY_COUNT", "20"))

    ASK_AI_ENDPOINT: str = os.getenv("ASK_AI_ENDPOINT", "")
    ASK_AI_API_KEY: str = os.getenv("ASK_AI_API_KEY", "")
    ASK_AI_MODEL_NAME: str = os.getenv("ASK_AI_MODEL_NAME", "")
    ASK_AI_TIMEOUT_SECONDS: int = int(os.getenv("ASK_AI_TIMEOUT_SECONDS", "30"))


settings = Settings()


def validate_settings():
    missing = [
        name for name in ["DB_SERVER", "DB_NAME", "DB_USERNAME", "DB_PASSWORD"]
        if not getattr(settings, name)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def validate_ask_ai_settings():
    missing = [
        name for name in ["ASK_AI_ENDPOINT", "ASK_AI_API_KEY", "ASK_AI_MODEL_NAME"]
        if not getattr(settings, name)
    ]
    if missing:
        raise RuntimeError(f"Missing required Ask AI environment variables: {', '.join(missing)}")

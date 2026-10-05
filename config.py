import os
from dotenv import load_dotenv
load_dotenv()
def _parse_int_list(value: str) -> list:
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(int(part))
        except ValueError:
            continue
    return result


class Settings:
    DB_SERVER: str = os.getenv("DB_SERVER", "")
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USERNAME: str = os.getenv("DB_USERNAME", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_DRIVER: str = os.getenv("DB_DRIVER", "pymssql")

    PRODUCT_TABLE: str = os.getenv("REBA_PRODUCT_TABLE", "Reba_Product")

    FACTGL_TABLE: str = os.getenv("REBA_FACTGL_TABLE", "Reba_FactGL")
    BUDGET_TABLE: str = os.getenv("REBA_BUDGET_TABLE", "Reba_Budget")
    GL_ACCOUNT_TABLE: str = os.getenv("REBA_GL_ACCOUNT_TABLE", "Reba_GeneralLedgerAccount")
    GL_BOOK_TABLE: str = os.getenv("REBA_GL_BOOK_TABLE", "Reba_GeneralLedgerBook")

    FIN_PROPERTY_KEY_COLUMN: str = os.getenv("FIN_PROPERTY_KEY_COLUMN", "Property BizKey")
    ACTUAL_VALUE_COLUMN: str = os.getenv("ACTUAL_VALUE_COLUMN", "Dollar")
    BUDGET_VALUE_COLUMN: str = os.getenv("BUDGET_VALUE_COLUMN", "Budget")

    ACTUAL_BOOK_KEYS: list = _parse_int_list(os.getenv("ACTUAL_BOOK_KEYS", ""))
    BUDGET_BOOK_KEYS: list = _parse_int_list(os.getenv("BUDGET_BOOK_KEYS", ""))

    GL_ACCOUNT_KEY_COLUMN: str = os.getenv("GL_ACCOUNT_KEY_COLUMN", "GLAccountKey")
    GL_ACCOUNT_NAME_COLUMN: str = os.getenv("GL_ACCOUNT_NAME_COLUMN", "")
    GL_ACCOUNT_CATEGORY_COLUMN: str = os.getenv("GL_ACCOUNT_CATEGORY_COLUMN", "")
    GL_ACCOUNT_CODE_COLUMN: str = os.getenv("GL_ACCOUNT_CODE_COLUMN", "")
    GL_ACCOUNT_LEVEL2_COLUMN: str = os.getenv("GL_ACCOUNT_LEVEL2_COLUMN", "GL Account Level 2")
    GL_ACCOUNT_LEVEL3_COLUMN: str = os.getenv("GL_ACCOUNT_LEVEL3_COLUMN", "GL Account Level 3")

    INCLUDE_UNMAPPED_ACCOUNTS: bool = os.getenv("INCLUDE_UNMAPPED_ACCOUNTS", "false").strip().lower() in ("1", "true", "yes")

    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))
    FINANCIAL_CACHE_TTL_SECONDS: int = int(os.getenv("FINANCIAL_CACHE_TTL_SECONDS", "900"))
    FINANCIAL_CACHE_MAX_ENTRIES: int = int(os.getenv("FINANCIAL_CACHE_MAX_ENTRIES", "256"))

    ACTIVE_STATUSES: list = os.getenv("ACTIVE_STATUSES", "Active,Stabilizing,Lease-Up").split(",")

    DEFAULT_NEARBY_COUNT: int = int(os.getenv("DEFAULT_NEARBY_COUNT", "5"))
    MAX_NEARBY_COUNT: int = int(os.getenv("MAX_NEARBY_COUNT", "20"))
    RADIUS_RESULT_CAP: int = int(os.getenv("RADIUS_RESULT_CAP", "50"))

    ASK_AI_ENDPOINT: str = os.getenv("ASK_AI_ENDPOINT", "")
    ASK_AI_API_KEY: str = os.getenv("ASK_AI_API_KEY", "")
    ASK_AI_MODEL_NAME: str = os.getenv("ASK_AI_MODEL_NAME", "")
    ASK_AI_TIMEOUT_SECONDS: int = int(os.getenv("ASK_AI_TIMEOUT_SECONDS", "30"))
    ASK_AI_MAX_TOKENS: int = int(os.getenv("ASK_AI_MAX_TOKENS", "700"))

    SECRET_KEY: str = os.getenv("SECRET_KEY", "")

    SAVECHAT_URI: str = os.getenv("SAVECHAT_URI", "")
    SAVECHAT_TIMEOUT_SECONDS: int = int(os.getenv("SAVECHAT_TIMEOUT_SECONDS", "30"))


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


def validate_secret_key_settings():
    if not settings.SECRET_KEY:
        raise RuntimeError("Missing required environment variable: SECRET_KEY")


def validate_savechat_settings():
    if not settings.SAVECHAT_URI:
        raise RuntimeError("Missing required environment variable: SAVECHAT_URI")

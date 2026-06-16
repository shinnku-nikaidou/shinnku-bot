import os

from dotenv import load_dotenv

load_dotenv()


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


TOKEN = os.environ["TOKEN"]
NAME = os.environ["NAME"]
ADMIN_TELEGRAM_USER_ID = int(os.environ["ADMIN_TELEGRAM_USER_ID"])
IS_MAINTENANCE = get_bool_env("IS_MAINTENANCE")
LIST_OF_ADMINS = [ADMIN_TELEGRAM_USER_ID]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
MESSAGE_CONTEXT_DB_PATH = os.environ.get(
    "MESSAGE_CONTEXT_DB_PATH", "data/messages.sqlite3"
)
MESSAGE_CONTEXT_LIMIT = int(os.environ.get("MESSAGE_CONTEXT_LIMIT", "20"))
MESSAGE_CONTEXT_MAX_AGE_DAYS = int(os.environ.get("MESSAGE_CONTEXT_MAX_AGE_DAYS", "30"))
api_id = os.environ["api_id"]
api_hash = os.environ["api_hash"]

WEBHOOK = False
# The following configuration is only needed if you setted WEBHOOK to True #
WEBHOOK_OPTIONS = {
    "listen": "0.0.0.0",  # IP
    "port": 443,
    "url_path": TOKEN,  # This is recommended for avoiding random people
    # making fake updates to your bot
    "webhook_url": f"https://example.com/{TOKEN}",
}

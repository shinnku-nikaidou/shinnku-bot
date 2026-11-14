"""Settings read from the environment (.env)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    tg_token: str
    owner_id: int
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    # Inner monologue before acting. Off sends thinking={"type": "disabled"}.
    llm_thinking: bool = True
    llm_timeout_s: float = 60.0
    # Model that plays the judge in the blind test. Empty means the chat model.
    llm_model_judge: str = ""
    # Model for the few merging calls of distillation. Empty means the chat model.
    llm_model_slow: str = ""
    # Same gateway; answers the bot's calls for a while after the chat model fails
    # (quota gone, gateway down), and always looks at pictures. Empty means no
    # backup, and the chat model looks at pictures.
    llm_model_fallback: str = ""
    # Saying one of these names in a chat wakes the bot up.
    name_triggers: tuple[str, ...] = ("真红", "shinnku")
    # Empty means every chat is allowed.
    allowed_chat_ids: frozenset[int] = frozenset()
    allow_private: bool = True
    timezone: str = "Asia/Shanghai"
    data_dir: Path = Path("data")
    assets_dir: Path = Path("assets")
    # Where the Telegram Desktop exports (result.json) live.
    export_dir: Path = Path("~/tg-export")
    # How much of the chat the model sees.
    context_messages: int = 30
    context_max_age_s: int = 6 * 3600
    # Local embedding model for recalling similar moments.
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    # How many of my past conversations sit in the fixed sample, and how many
    # similar moments are recalled per reply. A scene is about 89 tokens, so 520
    # scenes plus recall and the live chat make a prompt of roughly 50k tokens.
    anchor_scenes: int = 520
    recall_scenes: int = 24
    # Debounce: wait for the other person to stop typing.
    quiet_window_s: float = 2.5
    max_wait_s: float = 8.0
    # How many people cards and how many days of diary go into a reply.
    people_cards: int = 4
    diary_days: int = 3
    # Speaking without being called (only where /talk is above 0).
    yield_to_owner_min: int = 10
    proactive_calls_per_hour: int = 12
    proactive_calls_per_day: int = 150
    # Private per-reply trace (contains message text) under data_dir/trace.
    trace: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "live.sqlite3"

    @property
    def persona_dir(self) -> Path:
        return self.data_dir / "persona"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "trace"

    @property
    def corpus_db(self) -> Path:
        return self.data_dir / "corpus.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def distill_dir(self) -> Path:
        return self.data_dir / "distill"


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUE


def _csv(env: Mapping[str, str], name: str) -> list[str]:
    return [part.strip() for part in env.get(name, "").split(",") if part.strip()]


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env`, or from the process environment plus .env."""
    if env is None:
        load_dotenv()
        env = os.environ

    names = _csv(env, "NAME_TRIGGERS")
    defaults = Settings.__dataclass_fields__
    return Settings(
        tg_token=env["TOKEN"],
        owner_id=int(env["ADMIN_TELEGRAM_USER_ID"]),
        llm_base_url=env["LLM_BASE_URL"],
        llm_api_key=env["LLM_API_KEY"],
        llm_model=env["LLM_MODEL"],
        llm_thinking=_flag(env, "LLM_THINKING", True),
        llm_model_judge=env.get("LLM_MODEL_JUDGE", "").strip(),
        llm_model_slow=env.get("LLM_MODEL_SLOW", "").strip(),
        llm_model_fallback=env.get("LLM_MODEL_FALLBACK", "").strip(),
        name_triggers=tuple(names) if names else defaults["name_triggers"].default,
        allowed_chat_ids=frozenset(int(x) for x in _csv(env, "ALLOWED_CHAT_IDS")),
        allow_private=_flag(env, "ALLOW_PRIVATE", True),
        timezone=env.get("TIMEZONE", "").strip() or defaults["timezone"].default,
        data_dir=Path(env.get("DATA_DIR", "").strip() or "data"),
        assets_dir=Path(env.get("ASSETS_DIR", "").strip() or "assets"),
        export_dir=Path(env.get("EXPORT_DIR", "").strip() or "~/tg-export"),
        embed_model=env.get("EMBED_MODEL", "").strip() or defaults["embed_model"].default,
        anchor_scenes=int(env.get("ANCHOR_SCENES", "").strip() or defaults["anchor_scenes"].default),
        recall_scenes=int(env.get("RECALL_SCENES", "").strip() or defaults["recall_scenes"].default),
        trace=_flag(env, "TRACE", True),
        **{
            name: int(env.get(name.upper(), "").strip() or defaults[name].default)
            for name in (
                "people_cards",
                "diary_days",
                "yield_to_owner_min",
                "proactive_calls_per_hour",
                "proactive_calls_per_day",
            )
        },
    )

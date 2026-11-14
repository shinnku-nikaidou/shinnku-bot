"""Put a harness together from settings and whatever the corpus pipeline has built."""

from __future__ import annotations

from typing import Any

from shinnku_bot.config import Settings
from shinnku_bot.corpus.stats import limits_from_fingerprint, pace_from_fingerprint
from shinnku_bot.harness.engine import Harness
from shinnku_bot.harness.lookup import SearXNG
from shinnku_bot.harness.persona import Persona
from shinnku_bot.harness.recall import load_recall
from shinnku_bot.harness.send import Outbox
from shinnku_bot.harness.urge import Gate, load_urge
from shinnku_bot.llm import ChatModel, FallbackModel, GatewayModel
from shinnku_bot.store import Store


def gateway_model(settings: Settings, model: str = "", max_retries: int = 1) -> GatewayModel:
    return GatewayModel(
        settings.llm_base_url,
        settings.llm_api_key,
        model or settings.llm_model,
        thinking=settings.llm_thinking,
        timeout_s=settings.llm_timeout_s,
        max_retries=max_retries,
    )


def live_model(settings: Settings) -> ChatModel:
    """The chat model, and the backup behind it when one is set. Evaluations keep to the chat model.

    No SDK retries: the harness decides when to try again.
    """
    main = gateway_model(settings, max_retries=0)
    if not settings.llm_model_fallback:
        return main
    return FallbackModel(main, gateway_model(settings, settings.llm_model_fallback, max_retries=0))


def build_harness(
    settings: Settings,
    store: Store,
    outbox: Outbox,
    *,
    bot_id: int,
    bot_username: str | None = None,
    model: ChatModel | None = None,
    **kwargs: Any,
) -> Harness:
    """A harness with my measured style limits and recall, when those exist.

    Before the corpus is built it still works, on the placeholder persona alone.
    """
    recall = load_recall(
        settings.corpus_db, settings.index_dir, settings.embed_model, settings.models_dir
    )
    urge = load_urge(settings.persona_dir / "urge.json")
    gate = Gate(settings, urge) if urge is not None else None
    if model is None:
        model = live_model(settings)
        if settings.llm_model_fallback:
            # The chat model may not take pictures, and a refused picture would hand
            # every reply to the backup for a while; the backup looks at them instead.
            kwargs.setdefault("look_model", gateway_model(settings, settings.llm_model_fallback, max_retries=0))
    if settings.searxng_url:
        kwargs.setdefault("search", SearXNG(settings.searxng_url))
    return Harness(
        settings,
        store,
        model,
        outbox,
        Persona(settings.persona_dir),
        bot_id=bot_id,
        bot_username=bot_username,
        recall=recall,
        gate=gate,
        limits=limits_from_fingerprint(settings.persona_dir / "fingerprint.json"),
        pace=pace_from_fingerprint(settings.persona_dir / "fingerprint.json"),
        **kwargs,
    )

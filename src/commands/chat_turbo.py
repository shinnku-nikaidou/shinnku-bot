# encoding: utf-8
from logging import getLogger
from typing import Any

from openai import AsyncOpenAI
from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes
from telegramify_markdown import markdownify

from configurations import settings
from constants import ai
from utils.decorators import send_action
from utils.message_store import StoredMessage, build_reply_chain, record_message
from utils.text_handling import cut_command_text, replace_ai2shinnku

aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Init logger

logger = getLogger(__name__)


def _entity_mentions_bot(
    message: Message,
    entity: Any,
    bot_id: int,
    bot_username: str | None,
) -> bool:
    if entity.type == "text_mention":
        return bool(entity.user and entity.user.id == bot_id)
    if entity.type != "mention" or not bot_username:
        return False
    mention_text = message.text[entity.offset : entity.offset + entity.length]
    return mention_text.lstrip("@").lower() == bot_username.lower()


def remove_bot_mentions(
    message: Message,
    bot_id: int,
    bot_username: str | None,
) -> tuple[str, bool]:
    text: str = (message.text or "").strip()
    if not message.entities:
        return text, False

    mentioned = False
    for entity in sorted(message.entities, key=lambda item: item.offset, reverse=True):
        if not _entity_mentions_bot(message, entity, bot_id, bot_username):
            continue
        text = text[: entity.offset] + text[entity.offset + entity.length :]
        mentioned = True

    return text.strip(), mentioned


def should_reply_to_message(
    message: Message | None,
    bot_id: int,
    bot_username: str | None,
) -> bool:
    if message is None or message.text is None:
        return False
    if message.from_user is not None and message.from_user.is_bot:
        return False

    reply_to_bot = (
        message.reply_to_message is not None
        and message.reply_to_message.from_user is not None
        and message.reply_to_message.from_user.id == bot_id
    )
    _, mentioned = remove_bot_mentions(message, bot_id, bot_username)
    return reply_to_bot or mentioned


def _chain_to_model_input(
    chain: list[StoredMessage],
    bot_id: int,
    current_message_id: int,
    current_text: str,
) -> list[dict[str, str]]:
    model_input: list[dict[str, str]] = []
    for item in chain:
        text = current_text if item.message_id == current_message_id else item.text
        if not text:
            continue

        is_self = item.from_user_id == bot_id
        role = "assistant" if is_self else "user"
        name = "shinnku" if is_self else item.from_name
        model_input.append({"role": role, "content": f"{name}: {text}"})
    return model_input


def build_chat_context(
    message: Message, text: str, bot_id: int
) -> list[dict[str, str]]:
    chain = build_reply_chain(
        message.chat_id,
        message.message_id,
        limit=settings.MESSAGE_CONTEXT_LIMIT,
        max_age_days=settings.MESSAGE_CONTEXT_MAX_AGE_DAYS,
    )
    if not chain:
        return [{"role": "user", "content": text}]
    return _chain_to_model_input(chain, bot_id, message.message_id, text)


async def get_turbo_reply(model_input: list[dict[str, str]]) -> str:
    response = await aclient.responses.create(
        model=settings.OPENAI_MODEL,
        instructions=ai.prompt_shinnku,
        input=model_input,
    )
    return replace_ai2shinnku(response.output_text)


async def _reply_markdown_safe(message: Message, content: str) -> Message:
    formatted_content = markdownify(content)
    try:
        return await message.reply_text(
            formatted_content,
            parse_mode=ParseMode.MARKDOWN_V2,
        )
    except BadRequest as exc:
        if "can't parse entities" not in str(exc).lower():
            raise
        logger.warning("MarkdownV2 parse failed; falling back to plain text: %s", exc)
        return await message.reply_text(content)


async def _reply_with_context(
    message: Message,
    text: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> str:
    record_message(message)
    model_input = build_chat_context(message, text, context.bot.id)
    content = await get_turbo_reply(model_input)
    sent_message = await _reply_markdown_safe(message, content)
    record_message(sent_message, reply_to_message_id=message.message_id)
    return content


@send_action(ChatAction.TYPING)
async def chat_turbo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    message = update.message
    if message is None or message.text is None:
        return ""
    text = cut_command_text(message.text).strip()
    return await _reply_with_context(message, text, context)


async def chat_turbo_ref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    message = update.message
    bot_username = context.bot.username
    if not should_reply_to_message(message, context.bot.id, bot_username):
        return ""

    text, _ = remove_bot_mentions(message, context.bot.id, bot_username)
    return await _reply_with_context(message, text, context)

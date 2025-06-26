# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from utils.text_handling import cut_command_text, replace_ai2shinnku
from constants import ai

from configurations import settings

from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
import json

# Init logger

logger = getLogger(__name__)


async def get_turbo_reply(t: str):
    response = await aclient.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": ai.prompt_shinnku},
            {"role": "user", "content": t},
        ],
    )
    content = json.loads(response.choices[0].json())["message"]["content"]
    return replace_ai2shinnku(content)


@send_action(ChatAction.TYPING)
async def chat_turbo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    t = cut_command_text(update.message.text)
    content = await get_turbo_reply(t)
    await update.message.reply_text(content)


@send_action(ChatAction.TYPING)
async def chat_turbo_ref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    message = update.message
    text = message.text.strip()

    # Check if the current message is a reply to the bot
    reply_to_bot = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == context.bot.id
    )

    # Remove direct mentions of the bot and detect if one was present
    mentioned = False
    if message.entities:
        for entity in message.entities:
            if entity.type in ("mention", "text_mention"):
                mention_text = message.text[
                    entity.offset : entity.offset + entity.length
                ]
                if (
                    entity.type == "text_mention"
                    and entity.user
                    and entity.user.id == context.bot.id
                ) or (
                    entity.type == "mention"
                    and mention_text.lstrip("@") == context.bot.username
                ):
                    text = (
                        message.text[: entity.offset]
                        + message.text[entity.offset + entity.length :]
                    ).strip()
                    mentioned = True
                    break

    if not reply_to_bot and not mentioned:
        return

    content = await get_turbo_reply(text)
    await update.message.reply_text(content)

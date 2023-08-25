# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from utils.text_handling import cut_command_text, replace_ai2shinnku
from constants import ai

from configurations import settings

import openai
import json

# Init logger

logger = getLogger(__name__)


async def get_turbo_reply(t: str):
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = settings.OPENAI_API_KEY
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": ai.prompt_shinnku},
            {"role": "user", "content": t},
        ],
    )
    content = response["choices"][0]["message"]["content"]
    return replace_ai2shinnku(content)


@send_action(ChatAction.TYPING)
async def chat_turbo_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    t = cut_command_text(update.message.text)
    content = await get_turbo_reply(t)
    await update.message.reply_text(content)


@send_action(ChatAction.TYPING)
async def chat_turbo_ref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    content = await get_turbo_reply(update.message.text.strip())
    await update.message.reply_text(content)

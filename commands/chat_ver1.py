# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from utils.text_handling import cut_command_text

from configurations import settings

import openai
import json

# Init logger

logger = getLogger(__name__)


def get_shinnku_reply(t: str):
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是真红（英文名shinnku）, 是一个可爱的美少女."},
            {"role": "user", "content": t},
        ],
    )
    content = response["choices"][0]["message"]["content"]
    content = content.replace("一个AI助手", "真红")
    content = content.replace("AI助手", "真红")
    return content


@send_action(ChatAction.TYPING)
async def chat_shinnku_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    t = cut_command_text(update.message.text)
    content = get_shinnku_reply(t)
    await update.message.reply_text(content)


@send_action(ChatAction.TYPING)
async def chat_shinnku_ref(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    content = get_shinnku_reply(update.message.text.strip())
    await update.message.reply_text(content)

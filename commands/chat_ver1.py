# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from configurations import settings

import openai
import json

# Init logger

logger = getLogger(__name__)

openai.api_key = settings.OPENAI_API_KEY


@send_action(ChatAction.TYPING)
async def chat_shinnku(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    t = str(update.message.text[8:]).strip()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是真红（英文名shinnku）, 是一个可爱的美少女."},
            {"role": "user", "content": t},
        ],
    )
    response = json.loads(response)
    content = response["choices"][0]["message"]["content"]
    await update.message.reply_text(content)
    # await context.bot.send_message(
    #     chat_id=update.message.chat_id,
    #     text=xxx,
    # )

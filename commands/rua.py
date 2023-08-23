# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action

import random

# Init logger

logger = getLogger(__name__)


@send_action(ChatAction.TYPING)
async def rua(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    # pylint: disable=unused-argument
    sticker_id_1 = "CAACAgUAAx0CU2B2vQACBolk5FUDwuIQA1PPB7U6CP3zYb5gbQAC7ggAAvXI0FeiEnu0PWSsdjAE"  # 😧
    sticker_id_2 = "CAACAgUAAx0CU2B2vQACBo9k5FZ6DB6ZsFTsi1JgoDeob4kWFQACcwoAAlpg0FeY_SJn5kLnjjAE"  # 😫
    if random.random() < 0.5:
        await update.message.reply_sticker(sticker=sticker_id_1)
    else:
        await update.message.reply_sticker(sticker=sticker_id_2)

# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from constants import sticker
import random

# Init logger

logger = getLogger(__name__)


@send_action(ChatAction.TYPING)
async def rua(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    if random.random() < 0.5:
        await update.message.reply_sticker(sticker=sticker.shinnku[0])
    else:
        await update.message.reply_sticker(sticker=sticker.shinnku[1])

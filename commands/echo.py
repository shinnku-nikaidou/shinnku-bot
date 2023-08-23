# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action

# Init logger

logger = getLogger(__name__)


@send_action(ChatAction.TYPING)
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    await update.message.reply_text(update.message.text[6:])

# encoding: utf-8
from logging import getLogger

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import restricted
from utils.text_handling import cut_command_text

# Init logger

logger = getLogger(__name__)


@restricted
async def py(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    cmd = cut_command_text(update.message.text)
    try:
        a = eval(cmd)
    except Exception as e:
        a = e
    await update.message.reply_text(f"`{a}`", parse_mode="MarkdownV2")


@restricted
async def apy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    cmd = cut_command_text(update.message.text)
    try:
        a = await eval(cmd)
    except Exception as e:
        a = e
    await update.message.reply_text(f"`{a}`", parse_mode="MarkdownV2")

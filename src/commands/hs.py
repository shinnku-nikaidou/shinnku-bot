# encoding: utf-8
import json
from logging import getLogger
from urllib.parse import quote

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from utils.text_handling import cut_command_text

# Init logger

logger = getLogger(__name__)


@send_action(ChatAction.TYPING)
async def hs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Select an action: Adding parent/child or show data."""
    # pylint: disable=unused-argument
    args = cut_command_text(update.message.text)
    args = args.replace("（", "(")
    args = args.replace("）", ")")
    args = args.replace("，", ",")
    url = "https://tryhaskell.org/eval?exp="
    logger.debug(url + args)

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url + quote(args))

    if response.is_success:
        try:
            res = json.loads(response.text)["success"]
            await update.message.reply_text(text=res["type"])
            await update.message.reply_text(text=res["value"])
            return res["value"]
        except Exception:
            res = json.loads(response.text)["error"]
            await update.message.reply_text(text=res)
            return res

    await update.message.reply_text(text="unknown error")
    return "unknown error"

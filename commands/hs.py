# encoding: utf-8
from logging import getLogger
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from utils.decorators import send_action
from urllib.parse import quote
import json
import requests

# Init logger

logger = getLogger(__name__)


@send_action(ChatAction.TYPING)
async def hs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Select an action: Adding parent/child or show data."""
    # pylint: disable=unused-argument
    args = update.message.text.strip()[3:].strip()
    args = args.replace("（", "(")
    args = args.replace("）", ")")
    args = args.replace("，", ",")
    url = "https://tryhaskell.org/eval?exp="
    print(url + args)
    a = requests.get(url + quote(args), timeout=10)
    if a.ok:
        try:
            res = json.loads(a.text)["success"]
            await update.message.reply_text(
                text=res["type"],
            )
            await update.message.reply_text(
                text=res["value"],
            )
        except Exception:
            res = json.loads(a.text)["error"]
            await update.message.reply_text(
                text=res,
            )
    else:
        update.message.reply_text(
            text="unkown error",
        )

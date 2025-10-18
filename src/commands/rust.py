# encoding: utf-8
from logging import getLogger
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from utils.decorators import send_action
from utils.text_handling import cut_command_text

import json
import requests

# Init logger
logger = getLogger(__name__)

@send_action(ChatAction.TYPING)
async def rs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Evaluate Rust code using the Rust Playground API."""
    # pylint: disable=unused-argument
    # Normalize punctuation
    args = cut_command_text(update.message.text)
    args = args.replace("（", "(")
    args = args.replace("）", ")")
    args = args.replace("，", ",")

    url = "https://play.rust-lang.org/evaluate.json"
    payload = {
        "version": "stable",
        "optimize": "0",
        "code": args,
        "edition": "2021"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.ok:
            data = response.json()
            error = data.get("error")
            if error:
                await update.message.reply_text(text=error)
            else:
                await update.message.reply_text(text=data.get("result", ""))
        else:
            pass
            # await update.message.reply_text(text="Unknown error contacting Rust Playground")
    except Exception as e:
        logger.exception("Error evaluating Rust code")
        # await update.message.reply_text(text=str(e))

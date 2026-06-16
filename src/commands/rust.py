# encoding: utf-8
from logging import getLogger

import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from utils.decorators import send_action
from utils.text_handling import cut_command_text

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
        "edition": "2021",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload)
        if response.is_success:
            data = response.json()
            error = data.get("error")
            if error:
                await update.message.reply_text(text=error)
                return error
            result = data.get("result", "")
            await update.message.reply_text(text=result)
            return result
    except Exception:
        logger.exception("Error evaluating Rust code")
    return ""

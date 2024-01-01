from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import filters
from telegram.ext import MessageHandler

from commands.maintenance import maintenance
from commands.start import start
from commands.echo import echo
from commands.hs import hs
from commands.rua import rua, rua2
from commands.netease import netease
from commands.okiru import okiru
from commands.admin import py, apy, admin
from commands.ban import delete_msg, delete_pic
from commands.chat_turbo import chat_turbo_cmd, chat_turbo_ref
from configurations import settings
from configurations.settings import IS_MAINTENANCE
from utils import logger

if __name__ == "__main__":
    logger.init_logger(f"logs/{settings.NAME}.log")
    application = (
        Application.builder()
        .token(settings.TOKEN)
        .read_timeout(50)
        .write_timeout(50)
        .get_updates_read_timeout(50)
        .build()
    )
    print(f"IS_MAINTENANCE = {IS_MAINTENANCE}")
    if IS_MAINTENANCE:
        application.add_handler(CommandHandler("start", maintenance))
    else:
        application.add_handler(CommandHandler("start", start))
        application.add_handler(
            MessageHandler(
                filters.TEXT & filters.Regex(r"(啊(这|這)|(hai|海|🌊).*(⭐️|星|xin)|azkhx|quq)"),
                delete_msg,
            )
        )
        application.add_handler(CommandHandler("echo", echo))
        application.add_handler(CommandHandler("hs", hs))
        application.add_handler(CommandHandler("rua", rua))
        application.add_handler(CommandHandler("music", netease))
        application.add_handler(CommandHandler("okiru", okiru))
        application.add_handler(CommandHandler("chat", chat_turbo_cmd))
        application.add_handler(CommandHandler("py", py))
        application.add_handler(CommandHandler("apy", apy))
        application.add_handler(CommandHandler("admin", admin))
        application.add_handler(
            MessageHandler(
                ~filters.TEXT,
                rua2,
            )
        )
        application.add_handler(
            MessageHandler(
                ~filters.TEXT,
                delete_pic,
            )
        )

    application.run_polling()

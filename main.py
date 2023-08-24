from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import filters
from telegram.ext import MessageHandler

from commands.maintenance import maintenance
from commands.start import start
from commands.echo import echo
from commands.hs import hs
from commands.rua import rua
from commands.netease import netease
from commands.okiru import okiru
from commands.chat_ver1 import chat_shinnku
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
        # application.add_handler(
        #     MessageHandler(filters.TEXT & ~filters.COMMAND, maintenance))
    else:
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("echo", echo))
        application.add_handler(CommandHandler("hs", hs))
        application.add_handler(CommandHandler("rua", rua))
        application.add_handler(CommandHandler("netease", netease))
        application.add_handler(CommandHandler("okiru", okiru))
        application.add_handler(CommandHandler("shinnku", chat_shinnku))
        """YOUR COMMANDS IS HERE WHEN BOT IS NOT MAINTENANCE"""
    application.run_polling()

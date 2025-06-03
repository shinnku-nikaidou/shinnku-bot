from functools import wraps
from logging import getLogger

from configurations.settings import LIST_OF_ADMINS

logger = getLogger(__name__)


def restricted(func):
    @wraps(func)
    async def wrapped(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in LIST_OF_ADMINS:
            logger.warning("Unauthorized access denied for %s.", user_id)
            return
        return await func(update, context, *args, **kwargs)

    return wrapped


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        async def command_func(update, context, *args, **kwargs):
            await context.bot.send_chat_action(
                chat_id=update.effective_message.chat_id, action=action
            )
            return await func(update, context, *args, **kwargs)

        return command_func

    return decorator

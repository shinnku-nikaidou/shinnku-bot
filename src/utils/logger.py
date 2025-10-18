# encoding: utf-8
import logging
from logging.handlers import TimedRotatingFileHandler


def init_logger(logfile: str) -> logging.Logger:
    """Initialize the root logger and standard log handlers."""
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    rotate = TimedRotatingFileHandler(
        logfile,
        when="D",
        interval=1,
        backupCount=0,
    )
    rotate.setFormatter(log_formatter)
    root_logger.addHandler(rotate)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    return root_logger

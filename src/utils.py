import logging
import os


def set_loggings(level=logging.INFO, func_name=""):
    """
    TODO: set logging levels for object by overwriting the root logger
    """
    if isinstance(level, str):
        level = level.upper()
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = log_levels[level]
    # Remove all handlers associated with the root logger object:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,  # set logging level
        format="----- %(levelname)s (%(asctime)s) ----- \n%(message)s\n",
    )  # set messsage format
    logging.critical(
        "Hello %s, The current logging level is: %s",
        func_name,
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )

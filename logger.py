import logging
import transformers
import colorlog

has_init_logger = False
logger = None

logger_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def init_colorful_logger():
    global logger
    handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger("elaina")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def init_transformer_logger():
    from transformers.utils.logging import (
        enable_default_handler,
        enable_explicit_format,
        set_verbosity,
        INFO,
    )

    transformers.utils.logging.set_verbosity_info()
    set_verbosity(INFO)
    enable_default_handler()
    enable_explicit_format()

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )

if not has_init_logger:
    # init_logger()
    
    init_colorful_logger()
    init_transformer_logger()
    has_init_logger = True

# logger = logging.getLogger(__name__)




import logging
import transformers

has_init_logger = False


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    from transformers.utils.logging import (
        enable_default_handler,
        enable_explicit_format,
        set_verbosity,
        INFO
    )

    transformers.utils.logging.set_verbosity_info()
    set_verbosity(INFO)
    enable_default_handler()
    enable_explicit_format()

if not has_init_logger:
    init_logger()
    has_init_logger = True

# logger = logging.getLogger("elaina")
logger = logging.getLogger(__name__)

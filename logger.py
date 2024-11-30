# copy from https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/extras/logging.py
# and make some modifications
#
# Copyright 2024 Optuna, HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 文件说明，
# 该python在在get_logger时就已经创建好了一个logger,这个logger作为整个库的logger
# 被首先返回(只要你不传name,否则就会返回一个新的logger,但是这个logger没有做任何的设置)
# 该logger使用info_rank0函数时，logger输出的path不是你调用info_rank0的地方
# 而是 /root/elaina/utils/logging.py:127这样的
# 是你在logging.py中调用info_rank0中调用info的那一行

import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Optional

import colorlog

_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


class _Logger(logging.Logger):
    r"""
    A logger that supports info_rank0 and warning_once.
        这个类只有类型标注的作用，实际上返回的logger是logging.Logger
    """

    def info_rank0(self, *args, **kwargs) -> None:
        self.info(*args, **kwargs)

    def warning_rank0(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

    def warning_once(self, *args, **kwargs) -> None:
        self.warning(*args, **kwargs)

def _get_default_logging_level(config:dict = None) -> "logging._Level":
    r"""
    Returns the default logging level.
    获取默认的log level,
    优先读取配置信息
    """
    level_str = None
    if config:
        level_str = config.get('logger_level', None)
    if not level_str:
        level_str = os.environ.get("ELAINA_LOG_LEVEL", None)
    if level_str:
        if level_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[level_str.upper()]
        else:
            raise ValueError(f"Unknown logging level: {level_str}.")

    return _default_log_level



def _get_root_logger_name() -> str:
    # return __name__.split(".")[0]
    return "elaina"

def _get_library_root_logger() -> "_Logger":
    return logging.getLogger(_get_root_logger_name())

def _configure_library_root_logger() -> None:
    r"""
    Configures root logger using a stdout stream handler with an explicit format.
    """
    global _default_handler

    with _thread_lock:
        if _default_handler:  # already configured
            return

        # formatter = logging.Formatter(
        #     fmt="[%(levelname)s|%(asctime)s] %(name)s:%(lineno)s >> %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )
        # INFO|2024-11-30 01:26:55] utils:6 >> Hello, world!
        formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
            },
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False

def get_logger(name: Optional[str] = None) -> "_Logger":
    r"""
    Returns a logger with the specified name. It it not supposed to be accessed externally.
    """
    if name is None:
        name = _get_root_logger_name()

    _configure_library_root_logger()
    return logging.getLogger(name)

def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_once = warning_once


def init_transformer_logger():
    import transformers
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

logger = get_logger()

__all__ = ["get_logger", "init_transformer_logger", "logger"]
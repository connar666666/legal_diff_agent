"""统一日志配置。"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """配置根日志：控制台输出、可重复调用。"""
    root = logging.getLogger()
    if root.handlers:
        return logging.getLogger(name or __name__)

    lvl = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(lvl)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(lvl)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    root.addHandler(handler)

    return logging.getLogger(name or __name__)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

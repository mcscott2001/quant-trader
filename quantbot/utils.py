"""Small helpers shared by every part of the program."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml into a Python dictionary."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # make sure the working folders exist
    for folder in ("data", "logs", "reports", "models"):
        (ROOT / folder).mkdir(exist_ok=True)
    return cfg


def get_logger(name: str = "quantbot", logfile: str | None = None) -> logging.Logger:
    """A logger that prints to the screen and (optionally) to a file in logs/."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(ROOT / "logs" / logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def banner(logger: logging.Logger, step: int, total: int, title: str) -> None:
    logger.info("")
    logger.info("=" * 64)
    logger.info(f"  STEP {step}/{total}: {title}")
    logger.info("=" * 64)

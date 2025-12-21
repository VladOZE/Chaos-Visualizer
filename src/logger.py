"""
Утилиты для настройки логирования по модулям.

Каждый модуль может получить собственный логгер, который пишет в отдельный файл
в директорию ``logs``. Повторные вызовы возвращают уже созданный логгер.
"""

import logging
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str, filename: Optional[str] = None) -> logging.Logger:
    """
    Возвращает настроенный логгер.

    Args:
        name: Имя логгера (обычно имя модуля/подсистемы)
        filename: Имя файла лога (по умолчанию ``{name}.log``)

    Returns:
        logging.Logger: настроенный логгер
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Все модули пишут в один общий файл
    log_file = LOG_DIR / "app.log"

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


__all__ = ["get_logger", "LOG_DIR"]


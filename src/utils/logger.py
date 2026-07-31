"""
Logging configuration for the ML project.

Provides centralized logging setup used across
data pipelines and machine learning modules.

Author: Richard Obeng
"""

from pathlib import Path
import logging
import sys


def setup_logger(
    name: str = "ml_pipeline",
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return application logger.

    Parameters
    ----------
    name : str
        Logger name.

    log_file : str
        Path where logs will be stored.

    level : int
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Example
    -------
    logger = setup_logger()

    logger.info("Pipeline started")
    """

    logger = logging.getLogger(name)

    logger.setLevel(level)


    # Prevent duplicate handlers
    if logger.handlers:
        return logger


    # Create log directory
    log_path = Path(log_file)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Log format
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(filename)s:%(lineno)d | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


    # File handler
    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )

    file_handler.setLevel(level)

    file_handler.setFormatter(
        formatter
    )


    # Console handler
    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(level)

    console_handler.setFormatter(
        formatter
    )


    # Add handlers
    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )


    return logger



def get_logger(
    name: str = "ml_pipeline",
) -> logging.Logger:
    """
    Get existing project logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
    """

    return logging.getLogger(name)
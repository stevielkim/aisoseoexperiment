"""Logging utilities for geoseo_analysis package."""

import logging
import sys
from pathlib import Path


def setup_logging(name, level=logging.INFO, log_file=None):
    """
    Set up logging configuration.

    Parameters
    ----------
    name : str
        Logger name (usually __name__ or script name)
    level : int, optional
        Logging level, by default logging.INFO
    log_file : str or Path, optional
        Path to log file, by default None (console only)

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logging('my_script')
    >>> logger.info("Analysis started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

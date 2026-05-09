"""Logging system initialization and configuration."""

import sys
from pathlib import Path
from typing import cast

from loguru import logger
from tqdm import tqdm

from configs.app_config import LoggingConfig


class LoggerConfig:
    """Configure loguru logging system for the application."""

    def __init__(self, logging_config: LoggingConfig) -> None:
        """Initialize the logging system.

        Configures two handlers:
        1. tqdm handler: outputs logs to tqdm.write, avoiding conflicts with progress bars
        2. file handler: rotates log files by date, with auto-compression and cleanup.

        Configuration is read from APP_CONFIG.logging.
        """
        # Remove default stderr handler
        logger.remove()

        # Add tqdm-compatible handler (outputs logs above progress bar)
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

        # Create log directory if not exists
        log_dir = cast(Path, logging_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler (rotate by date, auto-cleanup of expired logs)
        logger.add(
            log_dir.joinpath(logging_config.file_pattern),
            retention=logging_config.retention,
            level=logging_config.level,
            colorize=True,
        )

        # Modify the global exception hook to log uncaught exceptions with loguru
        sys.excepthook = self.handle_exception

        logger.info(f"\nLogging system initialized, output directory: {log_dir}\n")

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions by logging them with loguru.

        Allows KeyboardInterrupt to exit gracefully without logging.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            "Uncaught exception occurred:"
        )

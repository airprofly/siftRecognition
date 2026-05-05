from pathlib import Path
import sys
from typing import cast

from loguru import logger
from tqdm import tqdm

from configs.app_config import LoggingConfig


class LoggerConfig:
    def __init__(self, logging_config: LoggingConfig) -> None:
        """Initialize the logging system.

        Configures two handlers:
        1. tqdm handler: outputs logs to tqdm.write, avoiding conflicts with progress bars
        2. file handler: rotates log files by date, with auto-cleanup of expired logs.

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

        # modify the global exception hook to log uncaught exceptions with loguru
        sys.excepthook = self.handle_exception

        logger.info(f"\nLogging system initialized, output directory: {log_dir}\n")

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Allow KeyboardInterrupt to exit gracefully without logging
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the uncaught exception with stack trace. Loguru's .opt(exception=...) will automatically extract stack info.
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical(
            "Uncaught exception occurred:"
        )

        # Call the default excepthook to ensure the program exits with a non-zero code and prints the traceback to stderr, but this will cause duplicate logging of the exception because the loguru has already logged it and then the default handler will print it again. 
        # sys.__excepthook__(exc_type, exc_value, exc_traceback) 


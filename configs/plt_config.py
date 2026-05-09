"""Matplotlib configuration for image visualization."""

import matplotlib.pyplot as plt
from loguru import logger


class PltConfig:
    """Configure matplotlib for dynamic image display during training."""

    def __init__(self) -> None:
        """Initialize matplotlib interactive mode and configure rcParams for image display."""
        plt.ion()
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(
            "\nMatplotlib plt.ion() enabled, images will display dynamically during training without blocking execution\n"
        )

"""Helper functions for optimizer configuration."""

import torch

from loguru import logger


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Returns the optimizer initialized according to the config.

    Args:
        model: The model to optimize for.
        config: A dictionary containing parameters for the config.

    Returns:
        The initialized optimizer.

    Raises:
        ValueError: If optimizer_type is not supported.
    """
    optimizer_type = config.get("optimizer_type", "sgd")
    learning_rate = config.get("lr", 1e-20)
    weight_decay = config.get("weight_decay", 1e3)

    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        logger.info("Using SGD optimizer")
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        logger.info("Using Adam optimizer")
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer

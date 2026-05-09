"""Deep learning utility functions for model prediction and loss computation."""

import torch


def predict_labels(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Perform the forward pass and extract the labels from the model output.

    Args:
        model: A PyTorch model that inherits from nn.Module.
        x: The input image tensor with shape (N, C, H, W).

    Returns:
        The predicted labels as a tensor of shape (N,).
    """
    output = model(x)
    predicted_labels = torch.argmax(output, dim=1)
    return predicted_labels


def compute_loss(
    model: torch.nn.Module,
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """Compute the loss between the model output and the target labels.

    Args:
        model: A PyTorch model with a loss_criterion attribute.
        model_output: The raw scores output by the network.
        target_labels: The ground truth class labels.
        is_normalize: If True, divide the loss by the batch size.

    Returns:
        The computed loss value as a scalar tensor.
    """
    loss = model.loss_criterion(model_output, target_labels)
    if is_normalize:
        loss = loss / model_output.shape[0]
    return loss

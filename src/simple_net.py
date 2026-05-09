"""Simple neural network for scene recognition."""

import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flatten layer to convert multi-dimensional input to 1D.

    This layer reshapes the input tensor from (N, C, H, W) to (N, C*H*W),
    which is commonly used between convolutional and fully connected layers.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten the input tensor.

        Args:
            x: Input tensor with shape (N, C, H, W).

        Returns:
            Flattened tensor with shape (N, C*H*W).
        """
        return x.view(x.size(0), -1)


class SimpleNet(nn.Module):
    """A simple 2-layer convolutional neural network for scene classification.

    The network consists of:
    - Two convolutional layers with ReLU activation and max pooling
    - Two fully connected layers for classification

    The network is designed for 64x64 grayscale input images and classifies
    them into 15 scene categories.
    """

    def __init__(self) -> None:
        """Initialize the SimpleNet layers and loss function.

        Note:
            The loss criterion uses 'sum' reduction, which means the loss is
            summed over all samples in the batch rather than averaged.
        """
        super(SimpleNet, self).__init__()

        # Convolutional layers
        self.cnn_layers = nn.Sequential(
            # Conv layer 1: 1 input channel, 10 output channels, 5x5 kernel
            nn.Conv2d(1, 10, kernel_size=5, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=3),  # [N, 10, 20, 20]
            nn.ReLU(),
            # Conv layer 2: 10 input channels, 20 output channels, 5x5 kernel
            nn.Conv2d(10, 20, kernel_size=5, stride=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=3),  # [N, 20, 5, 5]
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            Flatten(),
            # Input size: 5x5x20 = 500
            nn.Linear(500, 100),
            nn.Linear(100, 15),  # 15 scene categories
        )

        # Loss criterion with sum reduction
        self.loss_criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the network.

        Args:
            x: Input image tensor with shape (N, 1, H, W) where N is the
                batch size, 1 is the grayscale channel, and H, W are height
                and width (typically 64x64).

        Returns:
            Output tensor with shape (N, 15) containing raw scores for each
            of the 15 scene categories.
        """
        x = self.cnn_layers(x)  # [N, 20, 5, 5]
        x = self.fc_layers(x)   # [N, 15]
        return x

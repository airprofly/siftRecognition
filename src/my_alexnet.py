"""AlexNet-like architecture adapted for 64x64 grayscale scene recognition."""

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


class MyAlexNet(nn.Module):
    """AlexNet-like architecture adapted for 64x64 grayscale input.

    This network is inspired by the classic AlexNet architecture but adapted
    for smaller grayscale images (64x64 instead of 224x224 RGB). It features
    5 convolutional layers followed by 3 fully connected layers with dropout
    regularization.

    The network consists of:
    - 5 convolutional layers with ReLU activation and max pooling
    - 3 fully connected layers with dropout (p=0.5) for regularization
    """

    def __init__(self) -> None:
        """Initialize the MyAlexNet layers and loss function.

        Note:
            The loss criterion uses 'sum' reduction, which means the loss is
            summed over all samples in the batch rather than averaged.
        """
        super(MyAlexNet, self).__init__()

        # Convolutional layers (adapted from AlexNet)
        self.cnn_layers = nn.Sequential(
            # Conv layer 1: 1 -> 64 channels, 11x11 kernel, stride 4
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 64, 7, 7]

            # Conv layer 2: 64 -> 192 channels, 5x5 kernel
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 192, 2, 2]

            # Conv layer 3: 192 -> 384 channels, 3x3 kernel
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv layer 4: 384 -> 256 channels, 3x3 kernel
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv layer 5: 256 -> 256 channels, 3x3 kernel
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # [N, 256, 1, 1]
        )

        # Fully connected layers with dropout
        self.fc_layers = nn.Sequential(
            Flatten(),
            # Input size: 1x1x256 = 256
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 15),  # 15 scene categories
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
        x = self.cnn_layers(x)  # [N, 256, 1, 1]
        x = self.fc_layers(x)   # [N, 15]
        return x

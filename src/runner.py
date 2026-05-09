"""Trainer module for neural network training and evaluation."""

import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch.autograd import Variable

from .dl_util import compute_loss, predict_labels
from .image_loader import ImageLoader


class Trainer:
    """Trainer class for neural network training and evaluation.

    This class encapsulates the training loop, model saving/loading,
    loss/accuracy tracking, and visualization.
    """

    def __init__(
        self,
        data_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_dir: str,
        checkpoint_path: str,
        train_data_transforms: Optional[Callable],
        test_data_transforms: Optional[Callable],
        batch_size: int = 100,
        load_from_disk: bool = True,
        cuda: bool = False,
    ) -> None:
        """Initialize the Trainer with model, data, and training parameters.

        Args:
            data_dir: Directory containing the train/test data.
            model: The neural network model to train.
            optimizer: The optimizer for training.
            model_dir: Directory to save figures.
            checkpoint_path: Path to save/load model checkpoints.
            train_data_transforms: Transforms to apply to training data.
            test_data_transforms: Transforms to apply to test data.
            batch_size: Batch size for training and testing.
            load_from_disk: Whether to load existing checkpoint.
            cuda: Whether to use CUDA for GPU acceleration.
        """
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path

        self.model = model

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}

        self.train_dataset = ImageLoader(
            data_dir, split="train", transform=train_data_transforms
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.test_dataset = ImageLoader(
            data_dir, split="test", transform=test_data_transforms
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.optimizer = optimizer

        self.train_loss_history: list[float] = []
        self.validation_loss_history: list[float] = []
        self.train_accuracy_history: list[float] = []
        self.validation_accuracy_history: list[float] = []

        # Load the model from disk if checkpoint exists
        if load_from_disk and os.path.exists(self.checkpoint_path):
            map_location = "cuda" if cuda else "cpu"
            checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()

    def save_model(self) -> None:
        """Save the model state and optimizer state to disk.

        The checkpoint is saved to the specified checkpoint_path.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.checkpoint_path,
        )

    def train(self, num_epochs: int) -> None:
        """Run the main training loop for the specified number of epochs.

        Args:
            num_epochs: Number of epochs to train the model.
        """
        self.model.train()
        for epoch_idx in range(num_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                if self.cuda:
                    input_data, target_data = (
                        Variable(batch[0]).cuda(),
                        Variable(batch[1]).cuda(),
                    )
                else:
                    input_data, target_data = Variable(batch[0]), Variable(batch[1])

                output_data = self.model(input_data)
                loss = compute_loss(self.model, output_data, target_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.train_loss_history.append(loss.detach().item())
            self.model.eval()
            self.eval_on_test()
            self.validation_accuracy_history.append(self.get_accuracy(split="test"))
            self.train_accuracy_history.append(self.get_accuracy(split="train"))
            self.model.train()

            if epoch_idx % 1 == 0:
                print("Epoch:{}, Loss:{:.4f}".format(epoch_idx + 1, loss.detach().item()))
                # self.save_model()

    def eval_on_test(self) -> float:
        """Compute the loss on the test set.

        Returns:
            The average loss per example on the test set.
        """
        test_loss = 0.0

        num_examples = 0
        for batch_idx, batch in enumerate(self.test_loader):
            if self.cuda:
                input_data, target_data = (
                    Variable(batch[0]).cuda(),
                    Variable(batch[1]).cuda(),
                )
            else:
                input_data, target_data = Variable(batch[0]), Variable(batch[1])

            num_examples += input_data.shape[0]
            output_data = self.model.forward(input_data)
            loss = compute_loss(
                self.model, output_data, target_data, is_normalize=False
            )

            test_loss += loss.item()

        self.validation_loss_history.append(test_loss / num_examples)

        return self.validation_loss_history[-1]

    def get_accuracy(self, split: str = "test") -> float:
        """Get the accuracy on the specified dataset split.

        Args:
            split: Either "test" or "train" to select the dataset.

        Returns:
            The accuracy as a float between 0 and 1.
        """
        self.model.eval()

        num_examples = 0
        num_correct = 0
        for batch_idx, batch in enumerate(
            self.test_loader if split is "test" else self.train_loader
        ):
            if self.cuda:
                input_data, target_data = (
                    Variable(batch[0]).cuda(),
                    Variable(batch[1]).cuda(),
                )
            else:
                input_data, target_data = Variable(batch[0]), Variable(batch[1])

            num_examples += input_data.shape[0]
            predicted_labels = predict_labels(self.model, input_data)
            num_correct += torch.sum(predicted_labels == target_data).cpu().item()

        self.model.train()

        return float(num_correct) / float(num_examples)

    def plot_loss_history(self) -> None:
        """Plot the training and validation loss history."""
        fig = plt.figure()
        ep = range(len(self.train_loss_history))

        plt.plot(ep, self.train_loss_history, "-b", label="training")
        plt.plot(ep, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.model_dir, "loss_history.png"), dpi=150, bbox_inches="tight")
        logger.info(f"Loss history plot saved to {self.model_dir}/loss_history.png")
        plt.close(fig)

    def plot_accuracy(self) -> None:
        """Plot the training and validation accuracy history."""
        fig = plt.figure()
        ep = range(len(self.train_accuracy_history))
        plt.plot(ep, self.train_accuracy_history, "-b", label="training")
        plt.plot(ep, self.validation_accuracy_history, "-r", label="validation")
        plt.title("Accuracy history")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.model_dir, "accuracy.png"), dpi=150, bbox_inches="tight")
        logger.info(f"Accuracy plot saved to {self.model_dir}/accuracy.png")
        plt.close(fig)

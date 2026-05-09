"""Main entry point for scene recognition training.

This script automatically trains all models specified in configs/app_config.yml.
Each model's checkpoints and results are saved to separate directories.
"""

from pathlib import Path

import torch

from configs import APP_CONFIG
from loguru import logger
from src.data_transforms import get_data_augmentation_transforms, get_fundamental_transforms
from src.my_alexnet import MyAlexNet
from src.optimizer import get_optimizer
from src.runner import Trainer
from src.simple_net import SimpleNet
from src.simple_net_dropout import SimpleNetDropout
from src.stats_helper import compute_mean_and_std


def get_model(model_name: str):
    """Get model instance by name.

    Args:
        model_name: Name of the model ("simple", "dropout", or "alexnet").

    Returns:
        Model instance.
    """
    if model_name == "simple":
        return SimpleNet()
    elif model_name == "dropout":
        return SimpleNetDropout()
    elif model_name == "alexnet":
        return MyAlexNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main training function.

    Automatically trains all models specified in APP_CONFIG.training.models.
    Each model gets its own checkpoint directory under outputs/checkpoints/.
    """
    # Setup device from config
    device = APP_CONFIG.runtime.device
    if isinstance(device, str):
        use_cuda = device == "cuda" and torch.cuda.is_available()
    else:
        use_cuda = device.type == "cuda"
    actual_device = "cuda" if use_cuda else "cpu"
    logger.info(f"Config device: {device}, Using device: {actual_device}")

    # Compute dataset statistics once
    logger.info("Computing dataset mean and std...")
    dataset_mean, dataset_std = compute_mean_and_std(str(APP_CONFIG.paths.data_dir))
    logger.info(f"Dataset mean = {dataset_mean}, std = {dataset_std}")

    # Get input size from config
    inp_size = APP_CONFIG.model.input_size

    # Prepare data transforms
    train_transforms = get_data_augmentation_transforms(inp_size, dataset_mean, dataset_std)
    test_transforms = get_fundamental_transforms(inp_size, dataset_mean, dataset_std)

    # Get training parameters from config
    num_epochs = APP_CONFIG.training.num_epochs
    batch_size = APP_CONFIG.runtime.batch_size

    # Prepare optimizer config
    optimizer_config = {
        "optimizer_type": APP_CONFIG.optimizer.optimizer_type,
        "lr": APP_CONFIG.optimizer.lr,
        "weight_decay": APP_CONFIG.optimizer.weight_decay,
    }

    # Train each model specified in config
    models_to_train = APP_CONFIG.training.models
    logger.success(f"\nModels to train: {models_to_train}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {APP_CONFIG.optimizer.lr}\n")

    results = {}

    # Create checkpoints and figures directories
    checkpoint_dir = Path(APP_CONFIG.paths.output.checkpoint_dir)
    figures_dir = checkpoint_dir.parent / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models_to_train:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[START] Training {model_name} model")
        logger.info(f"{'=' * 60}\n")

        # Create model
        model = get_model(model_name)
        model_figure_dir = figures_dir / model_name
        model_figure_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pt"

        logger.info(f"Checkpoint path: {checkpoint_path}")
        logger.info(f"Figure directory: {model_figure_dir}")

        # Create optimizer
        optimizer = get_optimizer(model, optimizer_config)

        # Create trainer
        trainer = Trainer(
            data_dir=str(APP_CONFIG.paths.data_dir),
            model=model,
            optimizer=optimizer,
            model_dir=str(model_figure_dir),
            checkpoint_path=str(checkpoint_path),
            train_data_transforms=train_transforms,
            test_data_transforms=test_transforms,
            batch_size=batch_size,
            load_from_disk=False,
            cuda=use_cuda,
        )

        # Train model
        trainer.train(num_epochs=num_epochs)

        # Get final accuracies
        train_accuracy = trainer.train_accuracy_history[-1]
        validation_accuracy = trainer.validation_accuracy_history[-1]

        # Store results
        results[model_name] = {
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        }

        # Save model
        trainer.save_model()
        logger.success(f"[DONE] {model_name} model training completed")
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {validation_accuracy:.4f}")

        # Plot results
        logger.info("Plotting loss history and accuracy...")
        trainer.plot_loss_history()
        trainer.plot_accuracy()

    # Print summary of all results
    logger.success(f"\n{'=' * 60}")
    logger.success("[SUMMARY] All models trained successfully!")
    logger.success(f"{'=' * 60}\n")

    for model_name, result in results.items():
        log_msg = (
            f"{model_name:12s} - Train: {result['train_accuracy']:.4f} | "
            f"Validation: {result['validation_accuracy']:.4f}"
        )
        logger.info(log_msg)

    logger.success("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()

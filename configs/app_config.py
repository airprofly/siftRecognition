"""Application-level configuration for scene recognition project."""

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path

import torch
import yaml
from dacite import Config, from_dict


@dataclass(frozen=True)
class OutputConfig:
    """Manage output directory structure for model checkpoints and figures.

    Attributes:
        output_dir (str | Path): Output root directory, default ./outputs, supports string or Path object input.
        checkpoint_dir (Path): Pre-computed complete path for model checkpoints subdirectory with timestamp.
        figure_dir (Path): Pre-computed complete path for figures subdirectory with timestamp.
    """

    output_dir: str | Path = Path("./outputs")
    checkpoint_dir: str = "checkpoints"
    figure_dir: str = "figures"

    def __post_init__(self) -> None:
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # Pre-compute complete paths for subdirectories
        output_dir_path = Path(self.output_dir)
        object.__setattr__(self, "checkpoint_dir", output_dir_path.joinpath(self.checkpoint_dir))
        object.__setattr__(self, "figure_dir", output_dir_path.joinpath(self.figure_dir))


@dataclass(frozen=True)
class PathConfig:
    """Manage project input and output paths.

    Attributes:
        data_dir (str | Path): Dataset directory containing train/test splits, default ./data, supports string or Path object input.
        output (OutputConfig): Output directory structure and sub-paths configuration.
    """

    data_dir: str | Path = Path("./data")
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if isinstance(self.data_dir, str):
            object.__setattr__(self, "data_dir", Path(self.data_dir))


@dataclass(frozen=True)
class RuntimeConfig:
    """Manage runtime environment and training parameters.

    Attributes:
        device (str | torch.device): Computation device, default "cuda" with automatic fallback to CPU if CUDA unavailable.
        batch_size (int): Number of samples per batch, default 32, typical range 16-128 depending on GPU memory.
        num_workers (int): Number of data loading workers, default 1, increase for faster data loading on multi-core systems.
        pin_memory (bool): Whether to pin memory for faster GPU transfer, default True, only effective when using CUDA.
    """

    device: str | torch.device = "cuda"
    batch_size: int = 32
    num_workers: int = 1
    pin_memory: bool = True

    def __post_init__(self) -> None:
        # Convert device string to torch.device object with automatic CUDA availability check
        if isinstance(self.device, str):
            if self.device == "cuda" and torch.cuda.is_available():
                object.__setattr__(self, "device", torch.device("cuda"))
            else:
                object.__setattr__(self, "device", torch.device("cpu"))


@dataclass(frozen=True)
class ModelConfig:
    """Manage neural network architecture parameters.

    Attributes:
        input_size (tuple[int, int]): Input image dimensions as (height, width), default (64, 64).
        num_classes (int): Number of output classes for classification, default 15 for scene categories.
        dropout_rate (float): Dropout probability for regularization, default 0.5, range 0.0-1.0.
    """

    input_size: tuple[int, int] = (64, 64)
    num_classes: int = 15
    dropout_rate: float = 0.5


@dataclass(frozen=True)
class OptimizerConfig:
    """Manage optimizer configuration for model training.

    Attributes:
        optimizer_type (str): Optimizer algorithm, default "adam", options "sgd"/"adam".
        lr (float): Learning rate, default 1e-3, typical range 1e-5 to 1e-1.
        weight_decay (float): L2 regularization coefficient, default 1e-4, range 0.0 to 1e-2.
    """

    optimizer_type: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    """Manage training execution configuration.

    Attributes:
        models (list[str]): List of model names to train, options "simple"/"dropout"/"alexnet".
        num_epochs (int): Number of training epochs for each model.
    """

    models: list[str] = field(default_factory=lambda: ["simple", "dropout", "alexnet"])
    num_epochs: int = 30


@dataclass(frozen=True)
class LoggingConfig:
    """Manage logging behavior configuration.

    Attributes:
        log_dir (str | Path): Log output directory, default ./outputs, supports string or Path object input. Timestamp will be auto-added in load_from_yaml.
        level (str): Log level, default "INFO", options "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL".
        retention (str): Log file retention time, default "7 days", supports natural language format.
        file_pattern (str): Log filename pattern, default "{time:YYYY-MM-DD_HH-mm-ss}.log".
    """

    log_dir: str | Path = Path("./outputs")
    level: str = "INFO"
    retention: str = "7 days"
    file_pattern: str = "{time:YYYY-MM-DD_HH-mm-ss}.log"

    def __post_init__(self) -> None:
        if isinstance(self.log_dir, str):
            object.__setattr__(self, "log_dir", Path(self.log_dir))


@dataclass(frozen=True)
class AppConfig:
    """Aggregate global configuration entry point for scene recognition project.

    Attributes:
        paths (PathConfig): Path configuration for data and outputs.
        runtime (RuntimeConfig): Runtime environment and training parameters.
        model (ModelConfig): Neural network architecture parameters.
        optimizer (OptimizerConfig): Optimizer configuration.
        training (TrainingConfig): Training execution configuration (which models to train).
        logging (LoggingConfig): Logging-related configuration.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            AppConfig: Loaded configuration instance.

        Raises:
            ValueError: If YAML file is not found or has invalid format.
        """
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
                print(f"\n\033[1;92m[SUCCESS] Configuration file loaded successfully: {yaml_path}\033[0m\n")
        except yaml.YAMLError as e:
            raise ValueError(f"\033[1;91mInvalid YAML format in {yaml_path}: {e}\033[0m") from e
        except FileNotFoundError as e:
            raise ValueError(f"\033[1;91mConfiguration file not found: {yaml_path}\033[0m") from e

        # Generate unified timestamp for this run
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Add timestamp to output_dir in paths
        timestamped_output_dir = None
        if "paths" in config_dict and "output" in config_dict["paths"]:
            if "output_dir" in config_dict["paths"]["output"]:
                output_dir_value = config_dict["paths"]["output"]["output_dir"]
                timestamped_output_dir = Path(output_dir_value) / timestamp
                config_dict["paths"]["output"]["output_dir"] = str(timestamped_output_dir)

        # Add timestamp to log_dir
        if "logging" in config_dict and "log_dir" in config_dict["logging"]:
            log_dir_value = config_dict["logging"]["log_dir"]
            config_dict["logging"]["log_dir"] = str(Path(log_dir_value) / timestamp)

        # Use dacite for automatic recursive dataclass conversion
        return from_dict(cls, config_dict, Config(cast=[tuple]))


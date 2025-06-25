import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training configuration for SimSiam SSL training"""

    # Model Architecture
    backbone: str = "resnet18"  # Backbone architecture for SimSiam
    ssl_model: str = "simsiam"

    # Training Hyperfparameters
    batch_size: int = 512
    epochs: int = 800  # Number of training epochs
    lr: float = 0.06  # Base learning rate
    wd: float = 5e-4  # Weight decay for optimizer
    momentum: float = 0.9  # SGD momentum

    # Data Loading
    eval_batch_size: int = 2048  # Batch size for evaluation (KNN)

    # Training Options
    use_mixed_precision: bool = True  # Enable mixed precision training


@dataclass
class ExpConfig:
    """Experiment configuration for logging, checkpointing, and paths"""

    project_name: str = "simsiam-cifar10"  # W&B project name

    # Paths
    weight_path: str = "checkpoints/"  # Directory to save model weights
    data_path: str = "data/"  # Path to dataset

    # Resume training
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from

    # Logging Configuration
    use_wandb: bool = True  # Whether to use Weights & Biases
    eval_frequency: int = 10  # How often to run KNN evaluation (every N epochs)
    save_frequency: int = 10  # How often to save checkpoints (every N epochs)

    # Debug & Monitoring
    verbose: bool = False  # Enable verbose output

    num_workers: int = 0


def parse_args() -> tuple[TrainConfig, ExpConfig]:
    """
    Parse command line arguments and return TrainConfig and ExpConfig objects.

    Returns:
        tuple: (args, TrainConfig, ExpConfig) objects populated with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="SimSiam SSL Training Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model Architecture
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18"],
        help="Backbone architecture for SimSiam",
    )
    model_group.add_argument(
        "--ssl-model",
        type=str,
        default="simsiam",
        help="Backbone architecture for SimSiam",
    )

    # Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--batch-size", "-b", type=int, default=512, help="Training batch size"
    )
    train_group.add_argument(
        "--epochs", "-e", type=int, default=800, help="Number of training epochs"
    )
    train_group.add_argument(
        "--learning-rate", "-lr", type=float, default=0.06, help="Base learning rate"
    )
    train_group.add_argument("-wd", type=float, default=5e-4, help="Weight decay")
    train_group.add_argument("--momentum", "-m", type=float, default=0.9, help="SGD momentum")

    # Optimizer & Scheduler
    opt_group = parser.add_argument_group("Optimizer & Scheduler")
    opt_group.add_argument(
        "--optimizer",
        "-o",
        type=str,
        choices=["sgd", "adamw"],
        default="sgd",
        help="Optimizer type",
    )
    opt_group.add_argument(
        "--scheduler",
        "-s",
        type=str,
        choices=["cosine", "linear", "none"],
        default="cosine",
        help="Learning rate scheduler type",
    )

    # Data Loading
    data_group = parser.add_argument_group("Data Loading")
    data_group.add_argument(
        "--eval-batch-size", type=int, default=2048, help="Batch size for evaluation"
    )

    # Training Options
    train_opt_group = parser.add_argument_group("Training Options")
    train_opt_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    # Resume Options
    resume_group = parser.add_argument_group("Resume Training")
    resume_group.add_argument(
        "--resume-from-checkpoint",
        "-r",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    # Experiment Configuration
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--project-name",
        type=str,
        default="simsiam-cifar10",
        help="W&B project name",
    )
    exp_group.add_argument(
        "--weight-path",
        type=str,
        default="checkpoints/",
        help="Directory to save model weights",
    )
    exp_group.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to dataset",
    )
    exp_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    # Logging Configuration
    log_group = parser.add_argument_group("Logging Configuration")
    log_group.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    log_group.add_argument(
        "--eval-frequency",
        type=int,
        default=10,
        help="Run KNN evaluation every N epochs",
    )
    log_group.add_argument(
        "--save-frequency",
        type=int,
        default=20,
        help="Save checkpoints every N epochs",
    )

    # Debug & Monitoring
    debug_group = parser.add_argument_group("Debug & Monitoring")
    debug_group.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Create config objects
    train_config = TrainConfig(
        backbone=args.backbone,
        ssl_model=args.ssl_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        wd=args.wd,
        momentum=args.momentum,
        eval_batch_size=args.eval_batch_size,
        use_mixed_precision=args.mixed_precision,
    )

    exp_config = ExpConfig(
        project_name=args.project_name,
        weight_path=args.weight_path,
        data_path=args.data_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_wandb=args.use_wandb,
        eval_frequency=args.eval_frequency,
        save_frequency=args.save_frequency,
        verbose=args.verbose,
        num_workers=args.num_workers,
    )

    return train_config, exp_config


def get_exp_name(train_config: TrainConfig) -> str:
    """
    Generate experiment name based on SSL model and backbone architecture.

    Args:
        train_config: TrainConfig object containing model configuration

    Returns:
        str: Experiment name in format "{ssl_model}_{backbone}"

    Example:
        >>> config = TrainConfig(ssl_model="simsiam", backbone="resnet18")
        >>> get_exp_name(config)
        'simsiam_resnet18'
    """
    return f"{train_config.ssl_model}_{train_config.backbone}"

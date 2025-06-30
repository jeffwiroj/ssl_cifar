import argparse
from omegaconf import OmegaConf
import os

def get_args_from_yaml() -> OmegaConf:
    """
    Parses command-line arguments to get the path to a YAML config file,
    loads the file, and returns the configuration as a nested namespace object.
    """
    # 1. Set up the argument parser
    # This parser is only responsible for finding the config file path.
    parser = argparse.ArgumentParser(
        description="Load configuration from a YAML file."
    )
    parser.add_argument(
        '--path', '-p',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    config_path = args.path

    # 3. Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # 4. Load the YAML file using OmegaConf
    try:
        conf = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Error loading or parsing YAML file with OmegaConf: {e}")
        raise

    return conf.train, conf.experiment

def get_exp_name(train_config) -> str:
    """
    Generate experiment name based on SSL model and backbone architecture.

    Args:
        train_config: The training part of the config

    Returns:
        str: Experiment name in format "{ssl_model}_{backbone}"

    Example:
        >>> config = TrainConfig(ssl_model="simsiam", backbone="resnet18")
        >>> get_exp_name(config)
        'simsiam_resnet18'
    """
    return f"{train_config.ssl_model}_{train_config.backbone}"


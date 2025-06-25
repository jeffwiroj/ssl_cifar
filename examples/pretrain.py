import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ssl_cifar.config import ExpConfig, TrainConfig, get_exp_name, parse_args
from ssl_cifar.data.transformations import get_ssl_augmentations
from ssl_cifar.data.transformations.shared import get_dataloaders
from ssl_cifar.models.backbone import get_backbone
from ssl_cifar.models.ssl_models import get_ssl_model
from ssl_cifar.trainer.train_n_val import train_n_val


def load_configs_from_checkpoint(checkpoint_path: str) -> Tuple[TrainConfig, ExpConfig]:
    """Load TrainConfig and ExpConfig from checkpoint file"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading configs from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract configs from checkpoint
    train_config_dict = checkpoint.get("train_config")
    exp_config_dict = checkpoint.get("exp_config")

    if train_config_dict is None or exp_config_dict is None:
        raise ValueError("Checkpoint does not contain required config information")

    # Create config objects from dictionaries
    train_config = TrainConfig(**train_config_dict)
    exp_config = ExpConfig(**exp_config_dict)

    print("Successfully loaded configs from checkpoint")
    print(f"SSL Model: {train_config.ssl_model}, Backbone: {train_config.backbone}")
    print(f"Epochs: {train_config.epochs}, Batch Size: {train_config.batch_size}")

    return train_config, exp_config


def load_checkpoint(
    checkpoint_path: str,
    ssl_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.GradScaler],
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint and return training state"""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model state
    ssl_model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model state dict")

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Loaded optimizer state dict")

    # Load scheduler state
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print("Loaded scheduler state dict")

    # Load scaler state if available
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print("Loaded scaler state dict")

    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "acc": checkpoint.get("acc", 0.0),
        "wandb_run_id": checkpoint.get("wandb_run_id"),
        "train_config": checkpoint.get("train_config"),
        "exp_config": checkpoint.get("exp_config"),
    }


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    start_epoch = 0
    wandb_run_id = None

    tc, ec = parse_args()
    if ec.resume_from_checkpoint:
        tc, ec = load_configs_from_checkpoint(ec.resume_from_checkpoint)

    if ec.verbose:
        print(tc)
        print(ec)

    ### Init data
    ssl_augmentation = get_ssl_augmentations(tc.ssl_model)
    dataloader, train_loader, test_loader = get_dataloaders(tc, ssl_augmentation)

    ### Init backbone and models
    backbone = get_backbone(tc.backbone)
    ssl_model = get_ssl_model(model_name=tc.ssl_model, backbone=backbone)
    ssl_model = ssl_model.to(device)

    if "cuda" in device.type:
        ssl_model.compile()

    optimizer = torch.optim.SGD(
        params=ssl_model.parameters(), lr=tc.lr, weight_decay=tc.wd, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=len(dataloader) * tc.epochs
    )

    scaler = torch.GradScaler() if tc.use_mixed_precision else None

    # Load checkpoint if resuming
    if ec.resume_from_checkpoint is not None:
        checkpoint_state = load_checkpoint(
            ec.resume_from_checkpoint, ssl_model, optimizer, scheduler, scaler, device
        )
        start_epoch = checkpoint_state["start_epoch"]
        wandb_run_id = checkpoint_state["wandb_run_id"]
        print(f"Resuming training from epoch {start_epoch}")

    acc = train_n_val(
        ssl_model,
        optimizer,
        scheduler,
        scaler,
        dataloader,
        train_loader,
        test_loader,
        tc,
        ec,
        device,
        start_epoch=start_epoch,
        wandb_run_id=wandb_run_id
    ).item()

    if ec.weight_path:
        exp_name = get_exp_name(tc)
        checkpoint_path = os.path.join(ec.weight_path, f"{exp_name}_best.pth")
        os.makedirs(ec.weight_path, exist_ok=True)

        save_new_checkpoint = True

        if os.path.exists(checkpoint_path):
            prev_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            prev_acc = prev_checkpoint.get("final_acc", 0)

            print(acc, prev_acc)
            if acc <= prev_acc:
                save_new_checkpoint = False

        if save_new_checkpoint:
            torch.save(
                {"model_state_dict": ssl_model.state_dict(), "final_acc": acc, "config": tc},
                checkpoint_path,
            )

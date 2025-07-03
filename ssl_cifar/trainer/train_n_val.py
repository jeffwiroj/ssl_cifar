import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf

from ssl_cifar.models.knn import KNNClassifier


def train_n_val(
    ssl_model: torch.nn,
    optimizer,
    scheduler,
    scaler,
    dataloader,
    train_loader,
    test_loader,
    tc,
    ec,
    device="cpu",
    start_epoch: int = 0,
    wandb_run_id=None,
):
    """
    Trains a ssl model and evaluates its representations using a KNN classifier.

    Args:
        ssl_model: The self-supervised learning model (e.g., SimCLR, MoCo) with a backbone.
        optimizer: The optimizer used for training the SSL model.
        scheduler: The learning rate scheduler.
        dataloader: DataLoader for the training data (typically augmented pairs).
        train_loader: DataLoader for the training data to extract features for KNN.
        test_loader: DataLoader for the test data to evaluate KNN accuracy.
        tc: Training configuration.
        ec: Experiment configuration.
        device: The device to run training on ('cpu' or 'cuda').

    Returns:
        The final KNN accuracy on the test set.
    """

    wandb_run = None
    if ec.use_wandb:
        if wandb_run_id:
            wandb_run = wandb.init(
                project=ec.project_name, config=tc, id=wandb_run_id, resume="must"
            )
        else:
            wandb_run = wandb.init(project=ec.project_name, config=tc)
            wandb_run_id = wandb_run.id

    for epoch in range(start_epoch, tc.epochs):
        epoch_loss = 0.0
        accuracy = None
        for batch_idx, data in enumerate(dataloader):
            x1, x2 = data[0]
            x1 = x1.to(device)
            x2 = x2.to(device)

            current_lr = scheduler.get_last_lr()[0]

            optimizer.zero_grad()

            if tc.use_mixed_precision:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                ):
                    z1, loss = ssl_model(x1, x2)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                grads = [p.grad for p in ssl_model.parameters() if p.grad is not None]
                grad_norm = torch.nn.utils.get_total_norm(grads)
                if "max_norm" in tc:
                    torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), tc.max_norm)

                z_std = torch.std(z1, dim=0).mean()

                scaler.step(optimizer)
                scaler.update()
            else:
                z1, loss = ssl_model(x1, x2)

                # z1 shape: [B,2048]
                z_std = torch.std(z1, dim=0).mean()

                loss.backward()
                # Compute gradient norm before optimizer step
                grads = [p.grad for p in ssl_model.parameters() if p.grad is not None]
                grad_norm = torch.nn.utils.get_total_norm(grads)
                if "max_norm" in tc:
                    torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), tc.max_norm)
                optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if wandb_run:
                wandb_run.log(
                    {
                        "batch/loss": loss.item(),
                        "batch/learning_rate": current_lr,
                        "batch/gradient_norm": grad_norm,
                        "batch/z_std": z_std.item(),
                    }
                )

            # Optional: Print progress every N batches
            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{tc.epochs}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}, Grad Norm: {grad_norm:.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)

        # Log epoch-level metrics
        epoch_metrics = {
            "epoch/train_loss": avg_loss,
        }

        # Evaluate every 2 epochs to save time
        if (epoch + 1) % ec.eval_frequency == 0 or (epoch + 1) == tc.epochs:
            accuracy = evaluate(ssl_model, train_loader, test_loader, device)
            epoch_metrics["epoch/knn_accuracy"] = accuracy
            print(
                f"Epoch {epoch + 1}/{tc.epochs}, Avg Loss: {avg_loss:.4f}, "
                f"LR: {current_lr:.6f}, KNN Accuracy: {accuracy:.4f}"
            )
            if wandb_run:
                wandb_run.log(epoch_metrics)
        else:
            print(f"Epoch {epoch + 1}/{tc.epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        if ec.weight_path and (epoch + 1) % ec.save_frequency == 0:
            if accuracy is None:
                accuracy = evaluate(ssl_model, train_loader, test_loader, device)
            save_checkpoint(
                epoch=epoch,
                ssl_model=ssl_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                acc=accuracy,
                wandb_run_id=wandb_run.id if wandb_run else None,
                tc=tc,
                ec=ec,
            )

    if wandb_run:
        wandb.finish()

    return accuracy


def evaluate(ssl_model: nn.Module, train_loader, test_loader, device="cpu"):
    ssl_model.eval()  # Set to evaluation mode
    with torch.inference_mode():  # Disable gradient computation
        knn = KNNClassifier(backbone=ssl_model.backbone, device=device)  # Pass the trained backbone
        knn.fit(train_loader)
        _, accuracy = knn.predict(test_loader)
    ssl_model.train()  # Set back to training mode
    return accuracy


def save_checkpoint(
    epoch: int,
    ssl_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.GradScaler],
    acc: float,
    wandb_run_id: Optional[str],
    tc,
    ec,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": ssl_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "acc": acc,
        "train_config": OmegaConf.to_container(tc, resolve=True),
        "exp_config": OmegaConf.to_container(ec, resolve=True),
        "wandb_run_id": wandb_run_id,
    }

    os.makedirs(ec.weight_path, exist_ok=True)

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    exp_name = f"{tc.ssl_model}_{tc.backbone}"
    latest_path = os.path.join(ec.weight_path, f"{exp_name}_latest.pth")
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    checkpoint_path: str,
    ssl_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: str,
) -> Tuple[int, float, Optional[str], OmegaConf, OmegaConf]:
    """
    Loads model, optimizer, scheduler, and configs from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        ssl_model (nn.Module): The model instance to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        scheduler: The learning rate scheduler instance.
        scaler (Optional): The GradScaler instance for mixed precision.
        device (str): The device to map the loaded tensors to ('cuda' or 'cpu').

    Returns:
        A tuple containing:
        - start_epoch (int): The epoch to resume training from.
        - best_acc (float): The accuracy from the checkpoint.
        - wandb_run_id (Optional[str]): The wandb run ID for resuming logs.
        - tc (OmegaConf): The loaded training configuration.
        - ec (OmegaConf): The loaded experiment configuration.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

    # Load the checkpoint, mapping storage to the specified device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load states for model, optimizer, and scheduler
    ssl_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state if it exists in the checkpoint
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print("Loaded GradScaler state.")

    # Extract metadata
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint.get("acc", 0.0)  # Use .get for backward compatibility
    wandb_run_id = checkpoint.get("wandb_run_id")

    # --- Re-create OmegaConf objects from saved dictionaries ---
    # This is the reverse of OmegaConf.to_container()
    tc = OmegaConf.create(checkpoint["train_config"])
    ec = OmegaConf.create(checkpoint["exp_config"])

    print(f"Successfully loaded checkpoint from '{checkpoint_path}' at epoch {checkpoint['epoch']}")

    return {
        "start_epoch": start_epoch,
        "best_acc": best_acc,
        "wandb_run_id": wandb_run_id,
        "tc": tc,
        "ec": ec,
    }

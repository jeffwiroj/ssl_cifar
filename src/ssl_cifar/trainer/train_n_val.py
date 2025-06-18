from dataclasses import asdict

import torch
import torch.nn as nn
import wandb
from ssl_cifar.config import ExpConfig, TrainConfig
from ssl_cifar.models.knn import KNNClassifier


def train_n_val(
    ssl_model: torch.nn,
    optimizer,
    scheduler,
    dataloader,
    train_loader,
    test_loader,
    tc: TrainConfig,
    ec: ExpConfig,
    device="cpu",
):
    wandb_run = None
    if ec.use_wandb:
        wandb_run = wandb.init(project=ec.project_name, config=asdict(tc))

    for epoch in range(tc.epochs):
        epoch_loss = 0.0

        for batch_idx, data in enumerate(dataloader):
            x1, x2 = data[0]
            x1 = x1.to(device)
            x2 = x2.to(device)

            current_lr = scheduler.get_last_lr()[0]

            optimizer.zero_grad()

            z1, loss = ssl_model(x1, x2)

            # z1 shape: [B,2048]
            z_std = torch.std(z1, dim=0).mean()

            loss.backward()

            # Compute gradient norm before optimizer step
            grads = [p.grad for p in ssl_model.parameters() if p.grad is not None]
            grad_norm = torch.nn.utils.get_total_norm(grads)

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
        if (epoch + 1) % ec.eval_frequency == 0:
            accuracy = evaluate(ssl_model, train_loader, test_loader, device)
            epoch_metrics["epoch/knn_accuracy"] = accuracy
            print(
                f"Epoch {epoch + 1}/{tc.epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, KNN Accuracy: {accuracy:.4f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{tc.epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    if wandb_run:
        wandb.finish()


def evaluate(ssl_model: nn.Module, train_loader, test_loader, device="cpu"):
    ssl_model.eval()  # Set to evaluation mode
    with torch.inference_mode():  # Disable gradient computation
        knn = KNNClassifier(
            backbone=ssl_model.backbone, device=device
        )  # Pass the trained backbone
        knn.fit(train_loader)
        _, accuracy = knn.predict(test_loader)
    ssl_model.train()  # Set back to training mode
    return accuracy

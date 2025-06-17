from random import shuffle

import torch
import torch.nn as nn
import torchvision as tv
import wandb
from ssl_cifar.data.transformations.shared import test_transformation
from ssl_cifar.data.transformations.simsiam import simsiam_augmentation
from ssl_cifar.models.backbone import get_resnet
from ssl_cifar.models.knn import KNNClassifier
from ssl_cifar.models.simsiam import SimSiam
from torch.utils.data import DataLoader

# Initialize wandb
wandb.init(
    project="simsiam-cifar10",
    config={
        "epochs": 100,
        "batch_size": 1024,
        "learning_rate": 0.12,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "eval_batch_size": 2048,
        "scheduler": "CosineAnnealingLR"
    }
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Create evaluation datasets once
train_set = tv.datasets.CIFAR10(
    root="data/", train=True, download=True, transform=test_transformation
)
test_set = tv.datasets.CIFAR10(
    root="data/", train=False, download=True, transform=test_transformation
)
train_loader = DataLoader(train_set, batch_size=2048, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=2048, shuffle=False, drop_last=False)

# Pretraining data
unlabelled_set = tv.datasets.CIFAR10(
    root="data/", train=True, download=True, transform=simsiam_augmentation
)

dataloader = DataLoader(unlabelled_set, batch_size=1024, pin_memory=True, shuffle=True, drop_last=True)

backbone = get_resnet()
simsiam = SimSiam(backbone)
simsiam = simsiam.to(device)

# Watch the model with wandb
wandb.watch(simsiam, log="all", log_freq=10)

epochs = 100
max_steps = len(dataloader) * epochs

optimizer = torch.optim.SGD(
    params=simsiam.parameters(), lr=0.12, weight_decay=5e-4, momentum=0.9
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_steps)


def compute_gradient_norm(model):
    """Compute the gradient norm of the model parameters"""
    total_norm = 0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    return total_norm


def evaluate():
    simsiam.eval()  # Set to evaluation mode
    with torch.inference_mode():  # Disable gradient computation
        knn = KNNClassifier(backbone=simsiam.backbone, device=device)  # Pass the trained backbone
        knn.fit(train_loader)
        _, accuracy = knn.predict(test_loader)
    simsiam.train()  # Set back to training mode
    return accuracy


global_step = 0

for epoch in range(epochs):
    epoch_loss = 0.0
    
    for batch_idx, data in enumerate(dataloader):
        x1, x2 = data[0]
        x1 = x1.to(device)
        x2 = x2.to(device)

        current_lr = scheduler.get_last_lr()[0]

        optimizer.zero_grad()

        z1, loss = simsiam(x1, x2)

        # z1 shape: [B,2048]
        z_std = torch.std(z1, dim=0).mean()

        loss.backward()
        
        # Compute gradient norm before optimizer step
        grads = [p.grad for p in simsiam.parameters() if p.grad is not None]
        grad_norm = torch.nn.utils.get_total_norm(grads)
        
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        global_step += 1

        # Log batch-level metrics to wandb
        wandb.log({
            "batch/loss": loss.item(),
            "batch/learning_rate": current_lr,
            "batch/gradient_norm": grad_norm,
            "batch/z_std": z_std.item(),
        })

        # Optional: Print progress every N batches
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}, Grad Norm: {grad_norm:.4f}")

    avg_loss = epoch_loss / len(dataloader)
    
    # Log epoch-level metrics
    epoch_metrics = {
        "epoch/avg_loss": avg_loss,
        "epoch/learning_rate": current_lr,
        "epoch/number": epoch + 1
    }
    
    # Evaluate every 2 epochs to save time
    if (epoch + 1) % 2 == 0:
        accuracy = evaluate()
        epoch_metrics["epoch/knn_accuracy"] = accuracy
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, KNN Accuracy: {accuracy:.4f}")
    else:
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    # Log epoch metrics
    wandb.log(epoch_metrics)

# Finish the wandb run
wandb.finish()

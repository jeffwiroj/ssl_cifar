import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ssl_cifar.config import get_args_from_yaml
from ssl_cifar.data.transformations.shared import get_dataloaders
from ssl_cifar.models.model import CifarClassifier
from ssl_cifar.models.ssl_models import get_ssl_model
from ssl_cifar.trainer.scheduler import get_cosine_schedule_with_warmup


def extract_features(model, dataloader, device):
    """Extract features from backbone and holds them in memory.
    Only possible with small datasets like CIFAR10
    """
    model.eval()

    all_features = []
    all_labels = []

    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y
            features = model.backbone(x)
            all_features.append(features.cpu())
            all_labels.append(y)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    dataset = TensorDataset(all_features, all_labels)
    return dataset


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

    tc, ec = get_args_from_yaml()

    if ec.verbose:
        print(tc)
        print(ec)

    ### Init data
    train_loader, test_loader = get_dataloaders(tc, ssl_augmentation=None, supervised_only=True)

    ### Init backbone and models
    checkpoint = torch.load(ec.ssl_weight_path, weights_only=False, map_location="cpu")
    ssl_model = get_ssl_model(tc)
    ssl_model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze the backbone parameters
    for param in ssl_model.backbone.parameters():
        param.requires_grad = False

    # The final linear layer's parameters will still have requires_grad=True by default
    model = CifarClassifier(backbone=ssl_model.backbone, backbone_dim=tc.backbone_dim)
    model = model.to(device)

    train_dataset = extract_features(model, train_loader, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=tc.batch_size,
        pin_memory=True,
        num_workers=train_loader.num_workers,
    )

    test_dataset = extract_features(model, test_loader, device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=tc.eval_batch_size,
        pin_memory=True,
        num_workers=test_loader.num_workers,
    )

    model = torch.compile(model)
    model.fc.train()

    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=tc.lr, weight_decay=tc.wd, momentum=0.9, fused=True
    )

    total_steps = len(train_loader) * tc.epochs
    num_warmup = tc.num_warmup if "num_warmup" in tc else 0
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=tc.lr_warmup, num_training_steps=total_steps
    )

    if ec.verbose:
        print(
            f"Len tr (loader, dataset): ({len(train_loader)}, {len(train_dataset)})"
            f"test (loader, dtaset): ({len(test_loader)}, {len(test_dataset)})"
        )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(tc.epochs):
        # --- Training Phase ---
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)

            # Forward pass
            outputs = model.forward_linear(x_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x_train.size(0)

            # Calculate training accuracy
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += y_train.size(0)
            correct_train += (predicted_train == y_train).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc_train = 100 * correct_train / total_train

        print(
            f"Epoch [{epoch + 1}/{tc.epochs}] | "
            f"Training Loss: {epoch_loss:.4f} | "
            f"Training Accuracy: {epoch_acc_train:.2f}%"
        )

        # --- Testing Phase ---
        model.eval()  # Set the model to evaluation mode
        correct_test = 0
        total_test = 0

        with torch.inference_mode():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                outputs = model.forward_linear(x_test)
                _, predicted_test = torch.max(outputs.data, 1)

                total_test += y_test.size(0)
                correct_test += (predicted_test == y_test).sum().item()

        epoch_acc_test = 100 * correct_test / total_test
        print(f"Test Accuracy: {epoch_acc_test:.2f}%")
        print("-" * 30)

    print("Finished Training")

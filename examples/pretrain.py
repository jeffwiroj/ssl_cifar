import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ssl_cifar.config import get_args_from_yaml,get_exp_name
from ssl_cifar.data.transformations import get_ssl_augmentations
from ssl_cifar.data.transformations.shared import get_dataloaders
from ssl_cifar.models.backbone import get_backbone
from ssl_cifar.models.ssl_models import get_ssl_model
from ssl_cifar.trainer.train_n_val import train_n_val,load_checkpoint




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

    tc,ec = get_args_from_yaml()

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
    if ec.resume_from_checkpoint:
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
        wandb_run_id=wandb_run_id,
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

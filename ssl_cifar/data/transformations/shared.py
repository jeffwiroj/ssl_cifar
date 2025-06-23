"""
Common fucntions and vars to be shared across different papers
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from ssl_cifar.config import TrainConfig

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

test_transformation = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)


def unnormalize(image, mean, std):
    """
    Undo normalization of an image tensor.
    Args:
        image (torch.Tensor): The image tensor.
        mean (tuple): The mean values used for normalization.
        std (tuple): The std values used for normalization.

    Returns:
        torch.Tensor: The unnormalized image.
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add mean to undo normalization.
    return image


def visualize_images(images, labels, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
    """
    Visualizes images
    Args:
        images,labels: images after normalization
        mean (tuple): The mean values used for normalization.
        std (tuple): The std values used for normalization.
        num_images (int): The number of images to display.
    """

    num_images = images.size(0)
    # Unnormalize the images
    images = unnormalize(images, mean, std)

    # Convert the image tensor to a NumPy array (for plotting)
    images = images.numpy().transpose((0, 2, 3, 1))  # Shape (N, C, H, W) -> (N, H, W, C)
    images = np.clip(images, 0, 1)  # Clip values to be between 0 and 1 for displaying

    # Create a grid of images
    _, axes = plt.subplots(1, num_images, figsize=(15, 15))

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis("off")  # Hide axes
        ax.set_title(f"Label: {labels[i].item()}")  # Show the label

    plt.show()


class DoubleAugmentation:
    """Implements double augmentations"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return [self.transform(img), self.transform(img)]


def get_dataloaders(tc: TrainConfig, ssl_augmentation):
    # OPTIMIZATION 1: Use more efficient data loading
    # Create evaluation datasets once
    train_set = tv.datasets.CIFAR10(
        root="data/", train=True, download=True, transform=test_transformation
    )
    test_set = tv.datasets.CIFAR10(
        root="data/", train=False, download=True, transform=test_transformation
    )
    # OPTIMIZATION: Add num_workers and persistent_workers for faster data loading
    train_loader = DataLoader(
        train_set, batch_size=tc.eval_batch_size, shuffle=False, drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=tc.eval_batch_size, shuffle=False, drop_last=False, pin_memory=True
    )

    # Pretraining data
    unlabelled_set = tv.datasets.CIFAR10(
        root="data/", train=True, download=True, transform=ssl_augmentation
    )

    num_workers = min(8, os.cpu_count())

    dataloader = DataLoader(
        unlabelled_set,
        batch_size=tc.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=num_workers,
    )

    return dataloader, train_loader, test_loader

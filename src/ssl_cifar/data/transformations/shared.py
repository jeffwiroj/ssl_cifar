"""
Common fucntions and vars to be shared across different papers
"""

import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import v2
import torch

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

import torch
from ssl_cifar.data.transformations.shared import CIFAR_MEAN, CIFAR_STD, DoubleAugmentation
from torchvision.transforms import v2

transformations = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(size=32, scale=(0.2, 1)),
        v2.RandomHorizontalFlip(),
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1,
                )
            ],
            p=0.8,
        ),
        v2.RandomGrayscale(p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
)

simsiam_augmentation = DoubleAugmentation(transform=transformations)

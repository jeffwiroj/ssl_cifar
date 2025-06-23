import torch.nn as nn
import torchvision as tv


def get_resnet(model_name="resnet18"):
    if model_name == "resnet18":
        model = tv.models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )

        layers = list(model.children())[:3] + list(model.children())[4:-1]  # Remove initial maxpool

        backbone = list(layers) + [nn.Flatten()]
        return nn.Sequential(*backbone)

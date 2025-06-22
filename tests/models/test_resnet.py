import torch
from ssl_cifar.models.backbone import get_resnet

img = torch.rand((8, 3, 32, 32))


def test_resnet18():
    model = get_resnet("resnet18")
    output = model(img)

    assert output.shape == (8, 512)

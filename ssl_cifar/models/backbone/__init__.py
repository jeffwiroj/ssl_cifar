from ssl_cifar.models.backbone.resnet import get_resnet


def get_backbone(backbone="resnet18"):
    if backbone == "resnet18":
        return get_resnet(backbone)

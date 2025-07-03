from ssl_cifar.models.backbone.resnet import get_resnet
from ssl_cifar.models.backbone.mb_net import get_mobilenet

def get_backbone(backbone="resnet18"):
    if backbone == "resnet18":
        return get_resnet(backbone)
    elif backbone == "mobilenet":
        return get_mobilenet()


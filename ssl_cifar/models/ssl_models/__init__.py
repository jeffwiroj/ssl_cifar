from ssl_cifar.models.backbone import get_backbone
from ssl_cifar.models.ssl_models.barlow_twins import BarlowTwins
from ssl_cifar.models.ssl_models.simsiam import SimSiam


def get_ssl_model(tc):
    model_name = tc.ssl_model
    backbone_dim =   tc.backbone_dim if "backbone_dim" in tc else 512
    backbone = get_backbone(tc.backbone)

    if model_name == "simsiam":
        return SimSiam(backbone=backbone,backbone_dim=backbone_dim)
    elif model_name == "barlow_twins":
        return BarlowTwins(backbone=backbone, backbone_dim=backbone_dim,lambd=tc.lambd)

import timm
import torch.nn as nn

def get_mobilenet(model_name="v4"):
    if model_name=="v4":
        model = timm.create_model('mobilenetv4_conv_medium.e500_r256_in1k', pretrained=False)
        return nn.Sequential(*list(model.children())[:-1])
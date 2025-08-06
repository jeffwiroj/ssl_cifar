import torch
import torch.nn as nn

class CifarClassifier(nn.Module):
    
    def __init__(self, backbone,backbone_dim=512):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone_dim,10)
    
    def forward(self,x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.fc(features)
    
    def forward_linear(self,features):
        return self.fc(features)
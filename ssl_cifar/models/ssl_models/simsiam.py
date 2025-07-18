import torch.nn as nn


class SimSiam(nn.Module):
    def __init__(self, backbone,backbone_dim=512):
        super().__init__()
        self.criterion = nn.CosineSimilarity()
        self.backbone = backbone
        self.proj_mlp = nn.Sequential(
            nn.Linear(backbone_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
        )
        self.prediction_mlp = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2048),
        )

    def forward(self, x1, x2):
        z1, z2 = self.encode(x1), self.encode(x2)
        p1, p2 = self.prediction_mlp(z1), self.prediction_mlp(z2)

        loss = (
            -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2
        )

        return z1, loss

    def encode(self, x):
        return self.proj_mlp(self.backbone(x))

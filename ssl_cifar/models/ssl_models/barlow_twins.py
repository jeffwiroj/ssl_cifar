import torch
import torch.nn as nn


class BarlowTwins(nn.Module):
    def __init__(self, backbone, lambd, backbone_dim = 512,hid_dim=8192, out_dim=512):
        super().__init__()
        self.lambd = lambd
        self.backbone = backbone
        self.proj_mlp = nn.Sequential(
            nn.Linear(backbone_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim, bias=False),
        )

    def forward(self, x1, x2):
        # x of shape: [B,3,32,32]

        z1, z2 = self.encode(x1), self.encode(x2)  # shape [B,8192]
        z1 = (z1 - z1.mean(0, keepdim=True)) / (z1.std(0, keepdim=True) + 1e-8)
        z2 = (z2 - z2.mean(0, keepdim=True)) / (z2.std(0, keepdim=True) + 1e-8)
        B, D = z1.shape

        C = z1.T @ z2 / B
        diag_c = torch.diag(C)

        on_diag_loss = (diag_c - 1).pow(2).sum()
        off_diag_loss = (C.pow(2).sum() - diag_c.pow(2).sum()) * self.lambd

        loss = on_diag_loss + off_diag_loss

        return z1, loss

    def encode(self, x):
        return self.proj_mlp(self.backbone(x))

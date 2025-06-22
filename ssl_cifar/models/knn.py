import torch
import torch.nn.functional as F


class KNNClassifier:
    def __init__(
        self,
        backbone=None,
        k: int = 50,
        device: str = "cpu",
    ):
        self.k = k
        self.backbone = backbone
        self.device = device
        self.Xtr = None
        self.ytr = None

    def fit(self, dataloader):
        features = []
        labels = []
        with torch.inference_mode():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                feature = self.backbone(x)
                features.append(feature.cpu())
                labels.append(y.cpu())
        self.Xtr = torch.cat(features, dim=0)
        self.ytr = torch.cat(labels, dim=0)

    def predict(self, dataloader):
        if self.Xtr is None or self.ytr is None:
            print("Must call fit first!")
            return
        labels = []
        preds = []
        with torch.inference_mode():
            for x, y in dataloader:
                x = x.to(self.device)
                feature = self.backbone(x).cpu()
                distances = self._compute_distance(feature)
                _, top_inds = torch.topk(distances, k=self.k, dim=1, largest=False)
                nearest_labels = self.ytr[top_inds]  # [B,k]
                batch_preds = torch.mode(nearest_labels, dim=1)[0]
                preds.append(batch_preds)
                labels.append(y)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        accuracy = (preds == labels).sum() / labels.size(0)
        return preds, accuracy

    def _compute_distance(self, query):
        # Xtr shape, query shape: [N,512], [B,512]
        # Computes the 1 - cosine similarity
        query_normalized = F.normalize(query, dim=1)  # shape [B,512]
        Xtr_normalized = F.normalize(self.Xtr, dim=1)  # Shape [N,512]
        return 1 - (query_normalized @ Xtr_normalized.T)

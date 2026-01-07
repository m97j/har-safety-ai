import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128, pretrained=True, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, output_dim)

    def forward(self, x):  # (B,3,H,W)
        feat = self.backbone(x).flatten(1)
        return self.fc(feat)
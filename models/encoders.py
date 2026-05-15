import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=False):
        super().__init__()
        if name == "resnet18":
            if pretrained:
                try:
                    weights = models.ResNet18_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            else:
                weights = None
            net = models.resnet18(weights=weights)
            feat_dim = 512
        elif name == "resnet50":
            if pretrained:
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            else:
                weights = None
            net = models.resnet50(weights=weights)
            feat_dim = 2048
        elif name == "resnet34":
            if pretrained:
                try:
                    weights = models.ResNet34_Weights.IMAGENET1K_V1
                except AttributeError:
                    weights = "IMAGENET1K_V1"
            else:
                weights = None
            net = models.resnet34(weights=weights)
            feat_dim = 512
        else:
            raise ValueError("backbone must be resnet18, resnet34 or resnet50")
        self.feat_dim = feat_dim
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.pool = net.avgpool

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x

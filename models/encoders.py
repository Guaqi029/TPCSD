import os

import torch
import torch.nn as nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=False):
        super().__init__()
        self.name = name
        self.use_medclip = False

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

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.pool = net.avgpool

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

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.pool = net.avgpool

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

            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1
            self.layer2 = net.layer2
            self.layer3 = net.layer3
            self.layer4 = net.layer4
            self.pool = net.avgpool

        elif name == "medclip_vit":
            try:
                from medclip import MedCLIPModel, MedCLIPVisionModelViT
                from medclip import constants as medclip_constants
            except Exception as e:
                raise ImportError(
                    "medclip is required for backbone='medclip_vit'. "
                    "Install with `pip install medclip` or the MedCLIP GitHub package."
                ) from e

            model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            if pretrained:
                loaded = False
                try:
                    model.from_pretrained()
                    loaded = True
                except Exception:
                    pass

                if not loaded:
                    candidate_paths = []
                    env_path = os.environ.get("MEDCLIP_WEIGHTS_PATH", "").strip()
                    if env_path:
                        candidate_paths.append(env_path)
                    candidate_paths.append(
                        os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained", "medclip-resnet", "pytorch_model.bin")
                    )
                    candidate_paths.append(
                        os.path.join(os.path.expanduser("~"), ".medclip", medclip_constants.WEIGHTS_NAME)
                    )

                    ckpt_path = ""
                    for p in candidate_paths:
                        if p and os.path.isfile(p):
                            ckpt_path = p
                            break
                    if not ckpt_path:
                        raise FileNotFoundError(
                            "Could not find MedCLIP checkpoint. Set MEDCLIP_WEIGHTS_PATH or place checkpoint at "
                            "./pretrained/medclip-resnet/pytorch_model.bin"
                        )

                    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
                    model.load_state_dict(state_dict, strict=False)

            self.vision_model = model.vision_model
            self.use_medclip = True
            feat_dim = 512

        else:
            raise ValueError("backbone must be resnet18, resnet34, resnet50, or medclip_vit")

        self.feat_dim = feat_dim

    def forward(self, x):
        if self.use_medclip:
            out = self.vision_model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x

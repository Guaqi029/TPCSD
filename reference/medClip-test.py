#!/usr/bin/env python
"""
MedCLIP vision encoder sanity check (CPU/GPU compatible).

Usage:
  python reference/medclip-test.py

Optional env:
  MEDCLIP_WEIGHTS_PATH=/abs/path/to/pytorch_model.bin
"""


#  MedCLIP-ResNet50
#     load_mode: manual torch.load(map_location='cpu')
#   from /mnt/c/Users/guyiq/Desktop/kuosan-mrc/TPCSD/
#   pretrained/medclip-resnet/pytorch_model.bin
#     total_params: 24,556,608
#     trainable_params: 24,556,608
#     device: cpu
#     vision_output_shape: (2, 512)
#     forward_time_ms(batch=2): 175.91

#   MedCLIP-ViT
#     load_mode: manual torch.load(map_location='cpu')
#   from /mnt/c/Users/guyiq/Desktop/kuosan-mrc/TPCSD/
#   pretrained/medclip-resnet/pytorch_model.bin
#     total_params: 27,912,570
#     trainable_params: 27,912,570
#     device: cpu
#     vision_output_shape: (2, 512)
#     forward_time_ms(batch=2): 264.58

import os
import time
import torch

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import constants


def resolve_ckpt_path():
    # 1) explicit env path
    env_path = os.environ.get("MEDCLIP_WEIGHTS_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2) project-local path (your current setup)
    local_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "pretrained", "medclip-resnet", "pytorch_model.bin")
    )
    if os.path.isfile(local_path):
        return local_path

    # 3) medclip default cache
    cache_path = os.path.join(os.path.expanduser("~"), ".medclip", constants.WEIGHTS_NAME)
    if os.path.isfile(cache_path):
        return cache_path

    return ""


def load_medclip_cpu_safe(vision_cls):
    model = MedCLIPModel(vision_cls=vision_cls)

    # Try normal path first
    try:
        model.from_pretrained()
        return model, "from_pretrained()"
    except Exception as e:
        print(f"[warn] from_pretrained failed: {e}")

    ckpt_path = resolve_ckpt_path()
    if not ckpt_path:
        raise FileNotFoundError(
            "MedCLIP weights not found. Checked:\n"
            "  1) MEDCLIP_WEIGHTS_PATH\n"
            "  2) ./pretrained/medclip-resnet/pytorch_model.bin\n"
            "  3) ~/.medclip/pytorch_model.bin"
        )

    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[info] missing keys: {len(missing)}")
    if unexpected:
        print(f"[info] unexpected keys: {len(unexpected)}")
    return model, f"manual torch.load(map_location='cpu') from {ckpt_path}"


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def inspect(vision_cls, name):
    model, load_mode = load_medclip_cpu_safe(vision_cls)
    vision = model.vision_model

    total, trainable = count_params(vision)
    print(f"\n{name}")
    print(f"  load_mode: {load_mode}")
    print(f"  total_params: {total:,}")
    print(f"  trainable_params: {trainable:,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        t0 = time.time()
        img_emb = model.vision_model(x)
        dt = (time.time() - t0) * 1000

    shape = tuple(img_emb[0].shape) if isinstance(img_emb, (tuple, list)) else tuple(img_emb.shape)

    print(f"  device: {device}")
    print(f"  vision_output_shape: {shape}")
    print(f"  forward_time_ms(batch=2): {dt:.2f}")


if __name__ == "__main__":
    inspect(MedCLIPVisionModel, "MedCLIP-ResNet50")
    inspect(MedCLIPVisionModelViT, "MedCLIP-ViT")




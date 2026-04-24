import argparse
import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data.dataset import ISICDataset
from data.transforms import Transforms
from models import ResNetBackbone, Projector, l2_normalize
from utils.csv_utils import label_frame_to_int
from utils.metrics import (
    append_per_class_records,
    build_group_specs,
    compute_avg_metrics,
    compute_per_class_metrics,
    format_tail_lines,
    plot_loss_curve,
    plot_group_curves,
    update_loss_history,
    update_curve_history,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SingleTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img = self.transform(img)
        return img, label


def build_class_weights(train_csv, device):
    file = pd.read_csv(train_csv)
    labels = label_frame_to_int(file.iloc[:, 1:])
    counts = labels.sum(axis=0).to_numpy(dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = counts.max() / counts
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights, counts


def extract_features(encoder, projector, loader, device):
    encoder.eval()
    if projector is not None:
        projector.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            feat = encoder(img)
            if projector is not None:
                feat = projector(feat)
            feats.append(feat.detach().cpu())
            labels.append(label.detach().cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def compute_class_stats(feats, labels, num_classes):
    mu = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    sigma = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        f = feats[idx]
        mu[c] = f.mean(dim=0)
        sigma[c] = f.std(dim=0, unbiased=False)
    return mu, sigma


def compute_per_class_acc(classifier, feats, labels, num_classes, device):
    classifier.eval()
    with torch.no_grad():
        x = feats.to(device)
        y = labels.to(device)
        logits = classifier(x)
        pred = logits.argmax(dim=1)
    acc = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        idx = y == c
        if idx.sum() == 0:
            acc[c] = 0.0
        else:
            acc[c] = (pred[idx] == c).float().mean().item()
    return acc


def softmax_alloc(acc, alpha, total):
    scores = torch.exp(alpha * (1.0 - acc))
    weights = scores / scores.sum()
    counts = torch.floor(weights * float(total)).to(torch.long)
    # fix rounding to match total
    diff = int(total - counts.sum().item())
    if diff > 0:
        order = torch.argsort(weights, descending=True)
        for i in range(diff):
            counts[order[i % len(order)]] += 1
    return counts, weights


def sample_virtual_features(mu, sigma, prototypes, counts, gamma=0.2, delta=0.01):
    device = mu.device
    feats = []
    labels = []
    num_classes = mu.shape[0]
    for c in range(num_classes):
        k = int(counts[c].item())
        if k <= 0:
            continue
        eps = torch.randn(k, mu.shape[1], device=device)
        noise = torch.randn(k, mu.shape[1], device=device)
        mu_c = mu[c].unsqueeze(0)
        sigma_c = sigma[c].unsqueeze(0)
        proto_c = prototypes[c].unsqueeze(0)
        z = mu_c + eps * sigma_c + gamma * (proto_c - mu_c) + delta * noise
        feats.append(z)
        labels.append(torch.full((k,), c, device=device, dtype=torch.long))
    if not feats:
        return torch.empty(0, mu.shape[1], device=device), torch.empty(0, dtype=torch.long, device=device)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels


def evaluate_classifier(classifier, feats, labels, device, num_classes):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
    acc, f1, auc, bac, sens, spec = compute_avg_metrics(labels, probs)
    per_class = compute_per_class_metrics(labels, probs, num_classes=num_classes)
    return (acc, f1, auc, bac, sens, spec), per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ISIC2019LT")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--csv_file_train", required=True)
    parser.add_argument("--csv_file_val", required=True)
    parser.add_argument("--csv_file_test", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--log_dir", default="./log/tpcsd")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stage2_batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--use_projector", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=0)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--projector_ckpt", default="")
    parser.add_argument("--prototype_ckpt", default="")
    parser.add_argument("--lambda_mu", type=float, default=0.7)
    parser.add_argument("--gamma_proto", type=float, default=0.2)
    parser.add_argument("--delta_noise", type=float, default=0.01)
    parser.add_argument("--aas_alpha", type=float, default=1.0)
    parser.add_argument("--virtual_ratio", type=float, default=1.0)
    parser.add_argument("--merge_real", action="store_true")
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transforms = Transforms(args.image_size)
    train_base = ISICDataset(args.data_path, args.csv_file_train, transform=None)
    val_base = ISICDataset(args.data_path, args.csv_file_val, transform=None)
    test_base = ISICDataset(args.data_path, args.csv_file_test, transform=None)

    train_ds = SingleTransformDataset(train_base, transforms.test_transform)
    val_ds = SingleTransformDataset(val_base, transforms.test_transform)
    test_ds = SingleTransformDataset(test_base, transforms.test_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_base.n_class
    class_weights, counts = build_class_weights(args.csv_file_train, device)
    if not args.use_class_weight:
        class_weights = None
    group_specs = build_group_specs(train_base.class_names, counts)
    _, _, tail_classes = np.array_split(np.argsort(-counts), 3)
    tail_classes = [int(x) for x in tail_classes.tolist()]

    encoder = ResNetBackbone(args.backbone, pretrained=False)
    feat_dim = encoder.feat_dim
    projector = None
    proj_dim = feat_dim
    if args.use_projector:
        projector = Projector(feat_dim, proj_dim=args.proj_dim, hidden_dim=args.proj_hidden_dim)
        proj_dim = args.proj_dim

    encoder = encoder.to(device)
    if projector is not None:
        projector = projector.to(device)

    if not os.path.isfile(args.encoder_ckpt):
        raise FileNotFoundError(f"encoder_ckpt not found: {args.encoder_ckpt}")
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location="cpu"))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    if projector is not None and args.projector_ckpt:
        if not os.path.isfile(args.projector_ckpt):
            raise FileNotFoundError(f"projector_ckpt not found: {args.projector_ckpt}")
        projector.load_state_dict(torch.load(args.projector_ckpt, map_location="cpu"))
        projector.eval()
        for p in projector.parameters():
            p.requires_grad = False

    classifier = nn.Linear(proj_dim, num_classes).to(device)
    optimizer = optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    ckpt_root = os.path.join(args.checkpoints, args.run_name or f"run_tpcsd_stage2_{int(time.time())}")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{os.path.basename(ckpt_root)}.log")
    per_class_csv = os.path.join(ckpt_root, "per_class_metrics.csv")
    curve_history = {
        "train": {"epoch": [], "loss": []},
        "val": {"epoch": [], "acc": [], "bac": []},
        "test": {"epoch": [], "acc": [], "bac": []},
    }

    prototypes = None
    if args.prototype_ckpt and os.path.isfile(args.prototype_ckpt):
        prototypes = torch.load(args.prototype_ckpt, map_location="cpu").float()
    else:
        prototypes = torch.zeros(num_classes, proj_dim, dtype=torch.float32)

    for epoch in range(1, args.epochs + 1):
        # Extract real features
        train_feats, train_labels = extract_features(encoder, projector, train_loader, device)
        val_feats, val_labels = extract_features(encoder, projector, val_loader, device)
        test_feats, test_labels = extract_features(encoder, projector, test_loader, device)

        # Compute Gaussian stats
        mu, sigma = compute_class_stats(train_feats, train_labels, num_classes)
        mu = mu.to(device)
        sigma = sigma.to(device)

        proto = prototypes.to(device)
        if proto.shape[1] != mu.shape[1]:
            raise ValueError("prototype dimension mismatch with feature dim")
        mu_tilde = args.lambda_mu * mu + (1.0 - args.lambda_mu) * proto

        # AAS allocation
        per_class_acc = compute_per_class_acc(classifier, train_feats, train_labels, num_classes, device)
        total_virtual = int(len(train_labels) * float(args.virtual_ratio))
        alloc, weights = softmax_alloc(per_class_acc, args.aas_alpha, total_virtual)

        # Sample virtual features
        virt_feats, virt_labels = sample_virtual_features(
            mu_tilde, sigma, proto, alloc, gamma=args.gamma_proto, delta=args.delta_noise
        )

        if args.merge_real:
            feats = torch.cat([train_feats.to(device), virt_feats], dim=0)
            labels = torch.cat([train_labels.to(device), virt_labels], dim=0)
        else:
            feats = virt_feats
            labels = virt_labels

        # Train classifier on features
        classifier.train()
        dataset = TensorDataset(feats, labels)
        loader = DataLoader(dataset, batch_size=args.stage2_batch_size, shuffle=True, drop_last=False)
        loss_sum = 0.0
        for x, y in loader:
            logits = classifier(x)
            loss = nn.functional.cross_entropy(logits, y, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_metrics, val_per_class = evaluate_classifier(classifier, val_feats, val_labels, device, num_classes)
        test_metrics, test_per_class = evaluate_classifier(classifier, test_feats, test_labels, device, num_classes)

        train_loss = loss_sum / max(1, len(loader))
        append_per_class_records(per_class_csv, epoch, "val", val_per_class, train_base.class_names)
        append_per_class_records(per_class_csv, epoch, "test", test_per_class, train_base.class_names)
        update_loss_history(curve_history, epoch, train_loss)
        update_curve_history(curve_history, epoch, "val", val_per_class, group_specs)
        update_curve_history(curve_history, epoch, "test", test_per_class, group_specs)
        plot_loss_curve(curve_history, os.path.join(ckpt_root, "train_loss_curve.png"), title="Stage2 Train Loss")
        plot_group_curves(curve_history, os.path.join(ckpt_root, "val_group_curves.png"), "val")
        plot_group_curves(curve_history, os.path.join(ckpt_root, "test_group_curves.png"), "test")

        log_line = (
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.6f} "
            f"val_acc={val_metrics[0]:.6f} val_bac={val_metrics[3]:.6f} "
            f"test_acc={test_metrics[0]:.6f} test_bac={test_metrics[3]:.6f}"
        )
        print(log_line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
            for line in format_tail_lines(val_per_class, train_base.class_names, tail_classes, "val"):
                print(line)
                f.write(line + "\n")
            for line in format_tail_lines(test_per_class, train_base.class_names, tail_classes, "test"):
                print(line)
                f.write(line + "\n")

        torch.save(classifier.state_dict(), os.path.join(ckpt_root, "classifier_latest.pth"))
        torch.save(mu.cpu(), os.path.join(ckpt_root, "gaussian_mu_latest.pth"))
        torch.save(sigma.cpu(), os.path.join(ckpt_root, "gaussian_sigma_latest.pth"))


if __name__ == "__main__":
    main()

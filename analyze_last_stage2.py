import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.dataset import ISICDataset
from data.transforms import Transforms
from models import CosineClassifier, ResNetBackbone
from utils.checkpoint_utils import load_state_dict_flexible
from utils.csv_utils import label_frame_to_int
from utils.metrics import (
    build_group_specs,
    compute_avg_metrics,
    compute_macro_metric,
    compute_group_metric,
    compute_per_class_metrics,
    format_tail_lines,
    plot_loss_curve_from_log,
)
from visualize_embeddings import save_real_virtual_tsne, save_tsne_plot


class SingleTransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_counts(csv_file):
    frame = pd.read_csv(csv_file)
    labels = label_frame_to_int(frame.iloc[:, 1:])
    return labels.sum(axis=0).to_numpy(dtype=np.int64), list(frame.columns[1:])


def extract_features(encoder, loader, device):
    encoder.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            feat = encoder(image)
            feats.append(feat.detach().cpu())
            labels.append(label.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def compute_class_stats(feats, labels, num_classes):
    mu = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    sigma = torch.zeros(num_classes, feats.shape[1], dtype=feats.dtype)
    for class_id in range(num_classes):
        idx = labels == class_id
        if idx.sum() == 0:
            continue
        class_feats = feats[idx]
        mu[class_id] = class_feats.mean(dim=0)
        sigma[class_id] = class_feats.std(dim=0, unbiased=False)
    return mu, sigma


def compute_shared_covariance(feats, labels, mu, jitter=1e-4):
    if feats.numel() == 0:
        return torch.eye(mu.shape[1], dtype=mu.dtype)
    centered = feats - mu.index_select(0, labels.long())
    denom = max(int(centered.shape[0]) - 1, 1)
    cov = centered.t().matmul(centered) / float(denom)
    cov = 0.5 * (cov + cov.t())
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype) * float(jitter)
    return cov


def compute_per_class_acc(classifier, feats, labels, num_classes, device):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        pred = logits.argmax(dim=1).cpu()
    acc = torch.zeros(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        idx = labels == class_id
        if idx.sum() == 0:
            continue
        acc[class_id] = (pred[idx] == class_id).float().mean()
    return acc


def select_hardest_classes(per_class_acc, hardest_k, hardest_fraction):
    num_classes = int(per_class_acc.shape[0])
    if hardest_k > 0:
        k = min(num_classes, int(hardest_k))
    else:
        hardest_fraction = float(hardest_fraction)
        k = int(np.ceil(num_classes * hardest_fraction))
        k = max(1, min(num_classes, k))
    order = torch.argsort(per_class_acc, descending=False)
    return order[:k]


def allocate_hardest_virtual_counts(per_class_acc, hardest_indices, alpha, total):
    counts = torch.zeros_like(per_class_acc, dtype=torch.long)
    weights = torch.zeros_like(per_class_acc, dtype=torch.float32)
    total = int(total)
    if total <= 0 or hardest_indices.numel() == 0:
        return counts, weights

    hardest_acc = per_class_acc[hardest_indices]
    scores = torch.exp(float(alpha) * (1.0 - hardest_acc))
    hardest_weights = scores / scores.sum()
    hardest_counts = torch.floor(hardest_weights * float(total)).to(torch.long)
    diff = int(total - hardest_counts.sum().item())
    if diff > 0:
        order = torch.argsort(hardest_weights, descending=True)
        for i in range(diff):
            hardest_counts[order[i % len(order)]] += 1

    counts[hardest_indices] = hardest_counts
    weights[hardest_indices] = hardest_weights
    return counts, weights


def sample_virtual_features_shared_cov(mu_tilde, shared_cov, counts, delta=0.01, cov_scale_factor=1.0):
    feats = []
    labels = []
    base_cov = shared_cov * float(cov_scale_factor)
    base_cov = base_cov + torch.eye(shared_cov.shape[0], device=shared_cov.device, dtype=shared_cov.dtype) * (float(delta) ** 2)
    for class_id in range(mu_tilde.shape[0]):
        k = int(counts[class_id].item())
        if k <= 0:
            continue
        mvn = torch.distributions.MultivariateNormal(loc=mu_tilde[class_id], covariance_matrix=base_cov)
        z = mvn.sample((k,))
        feats.append(z)
        labels.append(torch.full((k,), class_id, dtype=torch.long, device=mu_tilde.device))
    if not feats:
        return torch.empty(0, mu_tilde.shape[1], device=mu_tilde.device), torch.empty(0, dtype=torch.long, device=mu_tilde.device)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def sample_virtual_features_diag(mu_tilde, sigma, prototypes, counts, gamma=0.2, delta=0.01):
    feats = []
    labels = []
    for class_id in range(mu_tilde.shape[0]):
        k = int(counts[class_id].item())
        if k <= 0:
            continue
        eps = torch.randn(k, mu_tilde.shape[1], device=mu_tilde.device)
        noise = torch.randn(k, mu_tilde.shape[1], device=mu_tilde.device)
        mu_c = mu_tilde[class_id].unsqueeze(0)
        sigma_c = sigma[class_id].unsqueeze(0)
        proto_c = prototypes[class_id].unsqueeze(0)
        z = mu_c + eps * sigma_c + gamma * (proto_c - mu_c) + delta * noise
        feats.append(z)
        labels.append(torch.full((k,), class_id, dtype=torch.long, device=mu_tilde.device))
    if not feats:
        return torch.empty(0, mu_tilde.shape[1], device=mu_tilde.device), torch.empty(0, dtype=torch.long, device=mu_tilde.device)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)




def summarize_gaussian_stats(mu, shared_cov):
    mu_norms = torch.norm(mu, p=2, dim=1)
    mu_norm_mean = float(mu_norms.mean().item())
    mu_norm_std = float(mu_norms.std(unbiased=False).item())

    cov_diag = torch.diag(shared_cov)
    cov_trace = float(cov_diag.sum().item())
    cov_diag_mean = float(cov_diag.mean().item())
    cov_fro = float(torch.norm(shared_cov, p="fro").item())

    eigvals = torch.linalg.eigvalsh(shared_cov)
    eig_min = float(eigvals.min().item())
    eig_max = float(eigvals.max().item())

    cond = float("inf")
    if eig_min > 1e-12:
        cond = eig_max / eig_min

    return {
        "mu_norm_mean": mu_norm_mean,
        "mu_norm_std": mu_norm_std,
        "cov_trace": cov_trace,
        "cov_diag_mean": cov_diag_mean,
        "cov_fro": cov_fro,
        "cov_eig_min": eig_min,
        "cov_eig_max": eig_max,
        "cov_cond": cond,
    }

def evaluate_classifier(classifier, feats, labels, device, num_classes):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
    avg_metrics = compute_avg_metrics(labels, probs)
    per_class_metrics = compute_per_class_metrics(labels, probs, num_classes=num_classes)
    return avg_metrics, per_class_metrics


def infer_classifier_head(classifier_ckpt, feature_dim, num_classes, default_cosine_scale):
    state = torch.load(classifier_ckpt, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"unsupported classifier checkpoint format: {classifier_ckpt}")

    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    if "bias" in state:
        return nn.Linear(feature_dim, num_classes).to(torch.device("cpu")), "linear"

    scale = float(default_cosine_scale)
    if "scale" in state:
        scale_value = state["scale"]
        scale = float(scale_value.item() if torch.is_tensor(scale_value) else scale_value)
    return CosineClassifier(feature_dim, num_classes, scale=scale).to(torch.device("cpu")), "cosine"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--csv_file_train", required=True)
    parser.add_argument("--csv_file_val", required=True)
    parser.add_argument("--csv_file_test", required=True)
    parser.add_argument("--encoder_ckpt", required=True)
    parser.add_argument("--classifier_ckpt", required=True)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--prototype_ckpt", default="")
    parser.add_argument("--gaussian_mu_ckpt", default="")
    parser.add_argument("--gaussian_sigma_ckpt", default="")
    parser.add_argument("--shared_cov_ckpt", default="")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aas_alpha", type=float, default=2.0)
    parser.add_argument("--virtual_ratio", type=float, default=1.0)
    parser.add_argument("--hardest_k", type=int, default=3)
    parser.add_argument("--hardest_fraction", type=float, default=0.5)
    parser.add_argument("--lambda_mu", type=float, default=0.5)
    parser.add_argument("--gamma_proto", type=float, default=0.2)
    parser.add_argument("--delta_noise", type=float, default=0.01)
    parser.add_argument("--cov_scale_factor", type=float, default=1.0)
    parser.add_argument("--cosine_scale", type=float, default=16.0)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    class_counts, class_names = build_class_counts(args.csv_file_train)
    groups = build_group_specs(class_names, class_counts)
    _, _, tail_classes = np.array_split(np.argsort(-class_counts), 3)
    tail_classes = [int(x) for x in tail_classes.tolist()]

    transforms = Transforms(args.image_size)
    train_base = ISICDataset(args.data_path, args.csv_file_train, transform=None)
    val_base = ISICDataset(args.data_path, args.csv_file_val, transform=None)
    test_base = ISICDataset(args.data_path, args.csv_file_test, transform=None)

    train_loader = DataLoader(
        SingleTransformDataset(train_base, transforms.test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SingleTransformDataset(val_base, transforms.test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        SingleTransformDataset(test_base, transforms.test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    encoder = ResNetBackbone(args.backbone, pretrained=False).to(device)
    load_state_dict_flexible(encoder, args.encoder_ckpt)
    feature_dim = encoder.feat_dim

    classifier, classifier_type = infer_classifier_head(
        args.classifier_ckpt, feature_dim, len(class_names), args.cosine_scale
    )
    classifier = classifier.to(device)
    load_state_dict_flexible(classifier, args.classifier_ckpt)

    train_feats, train_labels = extract_features(encoder, train_loader, device)
    val_feats, val_labels = extract_features(encoder, val_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)

    if args.gaussian_mu_ckpt and os.path.isfile(args.gaussian_mu_ckpt):
        mu = torch.load(args.gaussian_mu_ckpt, map_location="cpu").float()
    else:
        mu, _ = compute_class_stats(train_feats, train_labels, len(class_names))
    if args.shared_cov_ckpt and os.path.isfile(args.shared_cov_ckpt):
        shared_cov = torch.load(args.shared_cov_ckpt, map_location="cpu").float()
    else:
        shared_cov = compute_shared_covariance(train_feats, train_labels, mu)
    if args.gaussian_sigma_ckpt and os.path.isfile(args.gaussian_sigma_ckpt):
        sigma = torch.load(args.gaussian_sigma_ckpt, map_location="cpu").float()
    else:
        _, sigma = compute_class_stats(train_feats, train_labels, len(class_names))

    if args.prototype_ckpt and os.path.isfile(args.prototype_ckpt):
        prototypes = torch.load(args.prototype_ckpt, map_location="cpu").float()
    else:
        prototypes = torch.zeros(len(class_names), feature_dim, dtype=torch.float32)
    if prototypes.shape[1] != feature_dim:
        raise ValueError(
            "prototype dimension mismatch with encoder output dimension. "
            "Offline Stage2 analysis now expects projector-free Stage1 checkpoints."
        )

    mu = mu.to(device)
    sigma = sigma.to(device)
    shared_cov = shared_cov.to(device)

    stats = summarize_gaussian_stats(mu, shared_cov)
    print(
        "Gaussian stats: "
        f"mu_norm_mean={stats['mu_norm_mean']:.6f} "
        f"mu_norm_std={stats['mu_norm_std']:.6f} "
        f"cov_trace={stats['cov_trace']:.6f} "
        f"cov_diag_mean={stats['cov_diag_mean']:.6f} "
        f"cov_fro={stats['cov_fro']:.6f} "
        f"cov_eig_min={stats['cov_eig_min']:.6e} "
        f"cov_eig_max={stats['cov_eig_max']:.6e} "
        f"cov_cond={stats['cov_cond']:.6e}"
    )
    prototypes = prototypes.to(device)
    mu_tilde = args.lambda_mu * mu + (1.0 - args.lambda_mu) * prototypes

    val_metrics, val_per_class = evaluate_classifier(classifier, val_feats, val_labels, device, len(class_names))
    test_metrics, test_per_class = evaluate_classifier(classifier, test_feats, test_labels, device, len(class_names))
    train_acc = compute_per_class_acc(classifier, train_feats, train_labels, len(class_names), device)
    virtual_total = int(len(train_labels) * float(args.virtual_ratio))
    hardest_indices = select_hardest_classes(train_acc, args.hardest_k, args.hardest_fraction)
    alloc, alloc_weights = allocate_hardest_virtual_counts(train_acc, hardest_indices, args.aas_alpha, virtual_total)
    if args.shared_cov_ckpt and os.path.isfile(args.shared_cov_ckpt):
        virt_feats, virt_labels = sample_virtual_features_shared_cov(
            mu_tilde,
            shared_cov,
            alloc,
            delta=args.delta_noise,
            cov_scale_factor=args.cov_scale_factor,
        )
    else:
        virt_feats, virt_labels = sample_virtual_features_diag(
            mu_tilde, sigma, prototypes, alloc, gamma=args.gamma_proto, delta=args.delta_noise
        )

    quality_rows = []
    for class_id, class_name in enumerate(class_names):
        real_idx = train_labels == class_id
        virt_idx = virt_labels.cpu() == class_id if virt_labels.numel() > 0 else torch.zeros(0, dtype=torch.bool)
        real_class_feats = train_feats[real_idx]
        virt_class_feats = virt_feats.detach().cpu()[virt_idx] if virt_idx.numel() > 0 else torch.empty(0, feature_dim)

        proto = F.normalize(prototypes[class_id].detach().cpu().unsqueeze(0), dim=1)
        if virt_class_feats.shape[0] > 0:
            virt_norm = F.normalize(virt_class_feats, dim=1)
            proto_cos = torch.matmul(virt_norm, proto.t()).squeeze(1)
            virt_mean = virt_class_feats.mean(dim=0)
            virt_var = virt_class_feats.std(dim=0, unbiased=False).mean().item()
        else:
            proto_cos = torch.tensor([])
            virt_mean = torch.zeros(feature_dim)
            virt_var = float("nan")

        real_mean = real_class_feats.mean(dim=0) if real_class_feats.shape[0] > 0 else torch.zeros(feature_dim)
        real_var = (
            real_class_feats.std(dim=0, unbiased=False).mean().item() if real_class_feats.shape[0] > 0 else float("nan")
        )

        quality_rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "train_support": int(real_idx.sum().item()),
                "train_acc": float(train_acc[class_id].item()),
                "virtual_count": int(alloc[class_id].item()),
                "allocation_weight": float(alloc_weights[class_id].item()),
                "proto_cos_mean": float(proto_cos.mean().item()) if proto_cos.numel() > 0 else float("nan"),
                "proto_cos_std": float(proto_cos.std(unbiased=False).item()) if proto_cos.numel() > 0 else float("nan"),
                "real_var_mean": float(real_var),
                "virt_var_mean": float(virt_var),
                "mean_l2_diff": float(torch.norm(virt_mean - real_mean, p=2).item()) if virt_class_feats.shape[0] > 0 else float("nan"),
            }
        )

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.classifier_ckpt), "offline_analysis_stage2")
    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.dirname(args.classifier_ckpt)
    run_name = os.path.basename(run_dir)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    inferred_log_path = os.path.join(repo_root, "log", "tpcsd", f"{run_name}.log")

    per_class_rows = []
    for split_name, metrics in (("val", val_per_class), ("test", test_per_class)):
        for row in metrics:
            item = dict(row)
            item["split"] = split_name
            item["class_name"] = class_names[int(row["class_id"])]
            per_class_rows.append(item)
    pd.DataFrame(per_class_rows).to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
    pd.DataFrame(quality_rows).to_csv(os.path.join(output_dir, "virtual_feature_quality.csv"), index=False)
    torch.save({"features": train_feats, "labels": train_labels}, os.path.join(output_dir, "train_embeddings.pt"))
    torch.save({"features": val_feats, "labels": val_labels}, os.path.join(output_dir, "val_embeddings.pt"))
    torch.save({"features": test_feats, "labels": test_labels}, os.path.join(output_dir, "test_embeddings.pt"))
    if virt_feats.numel() > 0:
        torch.save({"features": virt_feats.detach().cpu(), "labels": virt_labels.detach().cpu()}, os.path.join(output_dir, "virtual_embeddings.pt"))

    save_tsne_plot(
        features=val_feats,
        labels=val_labels,
        class_names=class_names,
        out_path=os.path.join(output_dir, "val_tsne.png"),
        title=f"{args.dataset} Stage2 VAL t-SNE",
        max_points=4000,
        seed=args.seed,
    )
    save_tsne_plot(
        features=test_feats,
        labels=test_labels,
        class_names=class_names,
        out_path=os.path.join(output_dir, "test_tsne.png"),
        title=f"{args.dataset} Stage2 TEST t-SNE",
        max_points=4000,
        seed=args.seed,
    )
    if virt_feats.numel() > 0:
        save_real_virtual_tsne(
            real_features=train_feats,
            real_labels=train_labels,
            virtual_features=virt_feats.detach().cpu(),
            virtual_labels=virt_labels.detach().cpu(),
            class_names=class_names,
            out_path=os.path.join(output_dir, "train_real_virtual_tsne.png"),
            title=f"{args.dataset} Stage2 Train Real vs Virtual t-SNE",
            seed=args.seed,
        )
    plot_loss_curve_from_log(
        inferred_log_path,
        os.path.join(output_dir, "train_loss_curve.png"),
        title="Stage2 Train Loss",
    )

    summary = {
        "dataset": args.dataset,
        "encoder_ckpt": args.encoder_ckpt,
        "classifier_ckpt": args.classifier_ckpt,
        "prototype_ckpt": args.prototype_ckpt,
        "gaussian_mu_ckpt": args.gaussian_mu_ckpt,
        "gaussian_sigma_ckpt": args.gaussian_sigma_ckpt,
        "shared_cov_ckpt": args.shared_cov_ckpt,
        "classifier_type": classifier_type,
        "val": {
            "acc": float(val_metrics[0]),
            "f1": float(val_metrics[1]),
            "auc": float(val_metrics[2]),
            "bac": float(val_metrics[3]),
            "bacc": float(compute_macro_metric(val_per_class, metric_key="bacc")),
            "group_acc": compute_group_metric(val_per_class, groups, metric_key="acc"),
            "group_bac": compute_group_metric(val_per_class, groups, metric_key="bac"),
            "group_bacc": compute_group_metric(val_per_class, groups, metric_key="bacc"),
        },
        "test": {
            "acc": float(test_metrics[0]),
            "f1": float(test_metrics[1]),
            "auc": float(test_metrics[2]),
            "bac": float(test_metrics[3]),
            "bacc": float(compute_macro_metric(test_per_class, metric_key="bacc")),
            "group_acc": compute_group_metric(test_per_class, groups, metric_key="acc"),
            "group_bac": compute_group_metric(test_per_class, groups, metric_key="bac"),
            "group_bacc": compute_group_metric(test_per_class, groups, metric_key="bacc"),
        },
        "virtual_total": int(virtual_total),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        "val: "
        f"acc={val_metrics[0]:.6f}, f1={val_metrics[1]:.6f}, auc={val_metrics[2]:.6f}, "
        f"bac={val_metrics[3]:.6f}, bacc={compute_macro_metric(val_per_class, metric_key='bacc'):.6f}"
    )
    print(
        "test: "
        f"acc={test_metrics[0]:.6f}, f1={test_metrics[1]:.6f}, auc={test_metrics[2]:.6f}, "
        f"bac={test_metrics[3]:.6f}, bacc={compute_macro_metric(test_per_class, metric_key='bacc'):.6f}"
    )
    for line in format_tail_lines(val_per_class, class_names, tail_classes, "val"):
        print(line)
    for line in format_tail_lines(test_per_class, class_names, tail_classes, "test"):
        print(line)
    print("AAS allocation:")
    for row in quality_rows:
        print(
            f"{row['class_name']}: train_acc={row['train_acc']:.4f}, "
            f"virtual_count={row['virtual_count']}, proto_cos_mean={row['proto_cos_mean']:.4f}, "
            f"real_var={row['real_var_mean']:.4f}, virt_var={row['virt_var_mean']:.4f}"
        )
    if os.path.isfile(inferred_log_path):
        print(f"train_loss_curve: {os.path.join(output_dir, 'train_loss_curve.png')}")
    print(f"val_tsne: {os.path.join(output_dir, 'val_tsne.png')}")
    print(f"test_tsne: {os.path.join(output_dir, 'test_tsne.png')}")
    if virt_feats.numel() > 0:
        print(f"train_real_virtual_tsne: {os.path.join(output_dir, 'train_real_virtual_tsne.png')}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()

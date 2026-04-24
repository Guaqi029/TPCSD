import argparse
import faulthandler
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset

try:
    import cv2

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

faulthandler.enable(all_threads=True)

from data.dataset import ISICDataset
from data.transforms import Transforms
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

from models import ResNetBackbone, Projector, l2_normalize
from utils.losses import (
    balanced_softmax_loss,
    compute_batch_class_means,
    deferred_balanced_softmax_loss,
    ema_update_prototypes,
    enqueue_feature_queue,
    pcd_loss,
    recalibrate_prototypes,
    sp_kd_loss,
    var_preserve_loss,
)


class PairTransformDataset(Dataset):
    def __init__(self, base_dataset, pair_transform):
        self.base_dataset = base_dataset
        self.pair_transform = pair_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        strong, weak = self.pair_transform(img)
        return strong, weak, label


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(train_csv, device):
    file = pd.read_csv(train_csv)
    labels = label_frame_to_int(file.iloc[:, 1:])
    counts = labels.sum(axis=0).to_numpy(dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = counts.max() / counts
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return weights, counts, file.columns[1:]


def split_groups(counts):
    order = np.argsort(-counts)
    groups = np.array_split(order, 3)
    head = [int(i) for i in groups[0]]
    medium = [int(i) for i in groups[1]]
    tail = [int(i) for i in groups[2]]
    return head, medium, tail


def evaluate(encoder, classifier, loader, device, num_classes):
    encoder.eval()
    classifier.eval()
    gts = []
    preds = []
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)
            feat = encoder(img)
            logits = classifier(feat)
            probs = torch.softmax(logits, dim=1)
            gts.append(label)
            preds.append(probs)
    gts = torch.cat(gts, dim=0)
    preds = torch.cat(preds, dim=0)
    acc, f1, auc, bac, sens, spec = compute_avg_metrics(gts, preds)
    per_class = compute_per_class_metrics(gts, preds, num_classes=num_classes)
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--use_projector", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--proj_hidden_dim", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--cls_loss",
        choices=("ce", "balanced_softmax", "deferred_balanced_softmax"),
        default="ce",
    )
    parser.add_argument("--cls_warmup_epochs", type=int, default=150)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--pcd_weight", type=float, default=1.0)
    parser.add_argument("--pcd_temp", type=float, default=0.05)
    parser.add_argument("--spkd_weight", type=float, default=10.0)
    parser.add_argument("--var_weight", type=float, default=0.2)
    parser.add_argument("--var_beta", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--proto_momentum", type=float, default=0.96)
    parser.add_argument("--recal_interval", type=int, default=5)
    parser.add_argument("--recal_alpha", type=float, default=0.15)
    parser.add_argument("--tail_alpha", action="store_true")
    parser.add_argument("--queue_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.workers > 0:
        try:
            torch.multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    transforms = Transforms(args.image_size)
    train_base = ISICDataset(args.data_path, args.csv_file_train, transform=None)
    val_base = ISICDataset(args.data_path, args.csv_file_val, transform=None)
    test_base = ISICDataset(args.data_path, args.csv_file_test, transform=None)

    train_ds = PairTransformDataset(train_base, transforms)
    val_ds = SingleTransformDataset(val_base, transforms.test_transform)
    test_ds = SingleTransformDataset(test_base, transforms.test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    num_classes = train_base.n_class
    class_weights, counts, class_names = build_class_weights(args.csv_file_train, device)
    counts_tensor = torch.tensor(counts, device=device, dtype=torch.float32)
    if not args.use_class_weight:
        class_weights = None
    if args.cls_loss in {"balanced_softmax", "deferred_balanced_softmax"} and args.use_class_weight:
        raise ValueError("Balanced Softmax variants should not be combined with class-weighted CE.")

    head_classes, medium_classes, tail_classes = split_groups(counts)
    group_specs = build_group_specs(class_names, counts)
    tail_mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    tail_mask[tail_classes] = True

    encoder = ResNetBackbone(args.backbone, pretrained=args.pretrained)
    feat_dim = encoder.feat_dim
    classifier = nn.Linear(feat_dim, num_classes)
    projector = None
    proj_dim = feat_dim
    if args.use_projector:
        projector = Projector(feat_dim, proj_dim=args.proj_dim, hidden_dim=args.proj_hidden_dim)
        proj_dim = args.proj_dim

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    if projector is not None:
        projector = projector.to(device)

    encoder_teacher = ResNetBackbone(args.backbone, pretrained=args.pretrained).to(device)
    if projector is not None:
        projector_teacher = Projector(feat_dim, proj_dim=args.proj_dim, hidden_dim=args.proj_hidden_dim).to(device)
    else:
        projector_teacher = None

    encoder_teacher.load_state_dict(encoder.state_dict())
    if projector is not None:
        projector_teacher.load_state_dict(projector.state_dict())
    for p in encoder_teacher.parameters():
        p.requires_grad = False
    if projector_teacher is not None:
        for p in projector_teacher.parameters():
            p.requires_grad = False

    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        classifier = nn.DataParallel(classifier)
        if projector is not None:
            projector = nn.DataParallel(projector)
        encoder_teacher = nn.DataParallel(encoder_teacher)
        if projector_teacher is not None:
            projector_teacher = nn.DataParallel(projector_teacher)

    optim_params = list(encoder.parameters()) + list(classifier.parameters())
    if projector is not None:
        optim_params += list(projector.parameters())
    optimizer = optim.SGD(optim_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ckpt_root = os.path.join(args.checkpoints, args.run_name or f"run_tpcsd_{int(time.time())}")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{os.path.basename(ckpt_root)}.log")
    per_class_csv = os.path.join(ckpt_root, "per_class_metrics.csv")
    curve_history = {
        "train": {"epoch": [], "loss": []},
        "val": {"epoch": [], "acc": [], "bac": []},
        "test": {"epoch": [], "acc": [], "bac": []},
    }

    prototypes = torch.zeros(num_classes, proj_dim, device=device)
    student_queue = torch.zeros(args.queue_size, proj_dim, device=device)
    teacher_queue = torch.zeros(args.queue_size, proj_dim, device=device)
    label_queue = torch.full((args.queue_size,), -1, dtype=torch.long, device=device)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        classifier.train()
        if projector is not None:
            projector.train()

        loss_sum = 0.0
        cls_loss_sum = 0.0
        pcd_loss_sum = 0.0
        spkd_loss_sum = 0.0
        var_loss_sum = 0.0
        for strong, weak, label in train_loader:
            strong = strong.to(device, non_blocking=True)
            weak = weak.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            feat_s = encoder(strong)
            logits_s = classifier(feat_s)
            if projector is not None:
                z_s = projector(feat_s)
            else:
                z_s = feat_s

            with torch.no_grad():
                feat_t = encoder_teacher(weak)
                if projector_teacher is not None:
                    z_t = projector_teacher(feat_t)
                else:
                    z_t = feat_t

            if args.cls_loss == "balanced_softmax":
                cls_loss = balanced_softmax_loss(logits_s, label, counts_tensor)
            elif args.cls_loss == "deferred_balanced_softmax":
                cls_loss = deferred_balanced_softmax_loss(
                    logits_s, label, counts_tensor, epoch=epoch, warmup_epochs=args.cls_warmup_epochs
                )
            else:
                cls_loss = nn.functional.cross_entropy(logits_s, label, weight=class_weights)

            sample_weights = None
            if args.tail_alpha:
                nmax = float(counts_tensor.max())
                alpha = torch.sqrt(torch.clamp(nmax / torch.clamp(counts_tensor, min=1.0), min=1.0))
                sample_weights = alpha[label]

            valid_queue_mask = label_queue >= 0
            queue_student = student_queue[valid_queue_mask]
            queue_teacher = teacher_queue[valid_queue_mask]
            queue_labels = label_queue[valid_queue_mask]

            enqueue_feature_queue(student_queue, z_s, label_queue=label_queue, labels=label)
            enqueue_feature_queue(teacher_queue, z_t)

            z_s_norm = l2_normalize(z_s, dim=1)
            pcd = pcd_loss(z_s, label, prototypes, temperature=args.pcd_temp, sample_weights=sample_weights)
            spkd = sp_kd_loss(z_s, z_t, student_memory=queue_student, teacher_memory=queue_teacher)
            if queue_student.numel() > 0:
                z_var_all = torch.cat([z_s_norm, l2_normalize(queue_student, dim=1)], dim=0)
                label_var_all = torch.cat([label, queue_labels], dim=0)
            else:
                z_var_all = z_s_norm
                label_var_all = label
            var_loss = var_preserve_loss(z_var_all, label_var_all, head_classes, tail_classes, beta=args.var_beta)

            loss = cls_loss + args.pcd_weight * pcd + args.spkd_weight * spkd + args.var_weight * var_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            cls_loss_sum += cls_loss.item()
            pcd_loss_sum += args.pcd_weight * pcd.item()
            spkd_loss_sum += args.spkd_weight * spkd.item()
            var_loss_sum += args.var_weight * var_loss.item()

            # EMA update teacher
            with torch.no_grad():
                for t_param, s_param in zip(encoder_teacher.parameters(), encoder.parameters()):
                    t_param.data.mul_(args.ema_decay).add_(s_param.data, alpha=1.0 - args.ema_decay)
                if projector_teacher is not None and projector is not None:
                    for t_param, s_param in zip(projector_teacher.parameters(), projector.parameters()):
                        t_param.data.mul_(args.ema_decay).add_(s_param.data, alpha=1.0 - args.ema_decay)

            # Update prototypes
            with torch.no_grad():
                batch_class_mean, valid_mask = compute_batch_class_means(z_s.detach(), label, num_classes)
                prototypes = ema_update_prototypes(
                    prototypes, batch_class_mean, valid_mask, momentum=args.proto_momentum
                )
                if args.recal_interval > 0 and epoch % args.recal_interval == 0:
                    prototypes = recalibrate_prototypes(
                        prototypes,
                        batch_class_mean,
                        valid_mask,
                        tail_mask=tail_mask,
                        alpha=args.recal_alpha,
                    )

        train_loss = loss_sum / max(1, len(train_loader))
        train_cls_loss = cls_loss_sum / max(1, len(train_loader))
        train_pcd_loss = pcd_loss_sum / max(1, len(train_loader))
        train_spkd_loss = spkd_loss_sum / max(1, len(train_loader))
        train_var_loss = var_loss_sum / max(1, len(train_loader))
        val_metrics, val_per_class = evaluate(encoder, classifier, val_loader, device, num_classes)
        test_metrics, test_per_class = evaluate(encoder, classifier, test_loader, device, num_classes)
        val_acc, val_f1, val_auc, val_bac, val_sens, val_spec = val_metrics
        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec = test_metrics

        append_per_class_records(per_class_csv, epoch, "val", val_per_class, class_names)
        append_per_class_records(per_class_csv, epoch, "test", test_per_class, class_names)
        update_loss_history(curve_history, epoch, train_loss)
        update_curve_history(curve_history, epoch, "val", val_per_class, group_specs)
        update_curve_history(curve_history, epoch, "test", test_per_class, group_specs)
        plot_loss_curve(curve_history, os.path.join(ckpt_root, "train_loss_curve.png"), title="Stage1 Train Loss")
        plot_group_curves(curve_history, os.path.join(ckpt_root, "val_group_curves.png"), "val")
        plot_group_curves(curve_history, os.path.join(ckpt_root, "test_group_curves.png"), "test")

        log_line = (
            f"Epoch {epoch}/{args.epochs} "
            f"loss={train_loss:.6f} "
            f"loss_cls={train_cls_loss:.6f} "
            f"loss_pcd={train_pcd_loss:.6f} "
            f"loss_spkd={train_spkd_loss:.6f} "
            f"loss_var={train_var_loss:.6f} "
            f"val_acc={val_acc:.6f} val_bac={val_bac:.6f} "
            f"test_acc={test_acc:.6f} test_bac={test_bac:.6f}"
        )
        print(log_line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
            for line in format_tail_lines(val_per_class, class_names, tail_classes, "val"):
                print(line)
                f.write(line + "\n")
            for line in format_tail_lines(test_per_class, class_names, tail_classes, "test"):
                print(line)
                f.write(line + "\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(encoder.state_dict(), os.path.join(ckpt_root, "resnet_encoder_best.pth"))
            torch.save(classifier.state_dict(), os.path.join(ckpt_root, "classifier_best.pth"))
            if projector is not None:
                torch.save(projector.state_dict(), os.path.join(ckpt_root, "projector_best.pth"))
            torch.save(prototypes, os.path.join(ckpt_root, "prototype_memory_best.pth"))

        if args.cls_loss == "deferred_balanced_softmax":
            if args.cls_warmup_epochs > 1 and epoch == args.cls_warmup_epochs - 1:
                torch.save(encoder.state_dict(), os.path.join(ckpt_root, "resnet_encoder_pre_switch.pth"))
                torch.save(classifier.state_dict(), os.path.join(ckpt_root, "classifier_pre_switch.pth"))
                if projector is not None:
                    torch.save(projector.state_dict(), os.path.join(ckpt_root, "projector_pre_switch.pth"))
                torch.save(prototypes, os.path.join(ckpt_root, "prototype_memory_pre_switch.pth"))
            if epoch == args.cls_warmup_epochs:
                torch.save(encoder.state_dict(), os.path.join(ckpt_root, "resnet_encoder_switch_epoch.pth"))
                torch.save(classifier.state_dict(), os.path.join(ckpt_root, "classifier_switch_epoch.pth"))
                if projector is not None:
                    torch.save(projector.state_dict(), os.path.join(ckpt_root, "projector_switch_epoch.pth"))
                torch.save(prototypes, os.path.join(ckpt_root, "prototype_memory_switch_epoch.pth"))

        torch.save(encoder.state_dict(), os.path.join(ckpt_root, "resnet_encoder_latest.pth"))
        torch.save(classifier.state_dict(), os.path.join(ckpt_root, "classifier_latest.pth"))
        if projector is not None:
            torch.save(projector.state_dict(), os.path.join(ckpt_root, "projector_latest.pth"))
        torch.save(prototypes, os.path.join(ckpt_root, "prototype_memory_latest.pth"))


if __name__ == "__main__":
    main()

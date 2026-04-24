import torch
import torch.nn.functional as F


def balanced_softmax_loss(logits, labels, class_counts):
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    counts = torch.as_tensor(class_counts, dtype=logits.dtype, device=logits.device)
    counts = torch.clamp(counts, min=1.0)
    balanced_logits = logits + counts.log().unsqueeze(0)
    return F.cross_entropy(balanced_logits, labels)


def deferred_balanced_softmax_loss(logits, labels, class_counts, epoch, warmup_epochs):
    if int(epoch) < int(warmup_epochs):
        return F.cross_entropy(logits, labels)
    return balanced_softmax_loss(logits, labels, class_counts)


def enqueue_feature_queue(feature_queue, features, label_queue=None, labels=None):
    if feature_queue.numel() == 0 or features.numel() == 0:
        return

    capacity = int(feature_queue.shape[0])
    num_new = min(capacity, int(features.shape[0]))
    if num_new <= 0:
        return

    features = features.detach()[-num_new:]
    if label_queue is not None:
        if labels is None:
            raise ValueError("labels must be provided when label_queue is used")
        labels = labels.detach()[-num_new:]

    if num_new < capacity:
        feature_queue[:-num_new] = feature_queue[num_new:].clone()
        if label_queue is not None:
            label_queue[:-num_new] = label_queue[num_new:].clone()
    feature_queue[-num_new:] = features
    if label_queue is not None:
        label_queue[-num_new:] = labels


def sp_kd_loss(student_features, teacher_features, student_memory=None, teacher_memory=None, eps=1e-12):
    if student_features.numel() == 0:
        return student_features.new_tensor(0.0)

    if student_memory is not None and student_memory.numel() > 0:
        student_all = torch.cat([student_features, student_memory], dim=0)
    else:
        student_all = student_features

    if teacher_memory is not None and teacher_memory.numel() > 0:
        teacher_all = torch.cat([teacher_features, teacher_memory], dim=0)
    else:
        teacher_all = teacher_features

    gram_s = torch.matmul(student_features, student_all.t())
    gram_t = torch.matmul(teacher_features, teacher_all.t())
    gram_s = gram_s / torch.clamp(torch.norm(gram_s, p="fro"), min=eps)
    gram_t = gram_t / torch.clamp(torch.norm(gram_t, p="fro"), min=eps)
    return torch.sum((gram_s - gram_t) ** 2)


def pcd_loss(z, labels, prototypes, temperature=0.07, sample_weights=None):
    if z.numel() == 0:
        return z.new_tensor(0.0)
    z = F.normalize(z, p=2, dim=1)
    p = F.normalize(prototypes, p=2, dim=1)
    logits = torch.matmul(z, p.t()) / float(temperature)
    loss = F.cross_entropy(logits, labels, reduction="none")
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def var_preserve_loss(z, labels, head_classes, tail_classes, beta=0.8):
    if z.numel() == 0:
        return z.new_tensor(0.0)
    head_vars = []
    for c in head_classes:
        idx = labels == c
        if idx.sum() < 2:
            continue
        var = z[idx].var(dim=0, unbiased=False).mean()
        head_vars.append(var)
    if not head_vars:
        return z.new_tensor(0.0)
    head_var = torch.stack(head_vars).mean()

    tail_losses = []
    for c in tail_classes:
        idx = labels == c
        if idx.sum() < 2:
            continue
        var = z[idx].var(dim=0, unbiased=False).mean()
        tail_losses.append(F.relu(beta * head_var - var))
    if not tail_losses:
        return z.new_tensor(0.0)
    return torch.stack(tail_losses).mean()


def compute_batch_class_means(z, labels, num_classes):
    if z.numel() == 0:
        return z.new_zeros(num_classes, 0), torch.zeros(num_classes, dtype=torch.bool, device=z.device)

    batch_class_mean = z.new_zeros(num_classes, z.shape[1])
    valid_mask = torch.zeros(num_classes, dtype=torch.bool, device=z.device)
    for c in range(num_classes):
        idx = labels == c
        if idx.sum() == 0:
            continue
        batch_class_mean[c] = z[idx].mean(dim=0)
        valid_mask[c] = True
    return batch_class_mean, valid_mask


def ema_update_prototypes(prototypes, batch_class_mean, valid_mask, momentum=0.96):
    if batch_class_mean.numel() == 0 or not torch.any(valid_mask):
        return prototypes
    prototypes[valid_mask] = (
        momentum * prototypes[valid_mask] + (1.0 - momentum) * batch_class_mean[valid_mask]
    )
    return prototypes


def recalibrate_prototypes(prototypes, batch_class_mean, valid_mask, tail_mask=None, alpha=0.15, tail_factor=1.5):
    if batch_class_mean.numel() == 0 or not torch.any(valid_mask):
        return prototypes

    recal_factor = torch.ones(prototypes.shape[0], dtype=prototypes.dtype, device=prototypes.device)
    if tail_mask is not None:
        recal_factor = recal_factor + (float(tail_factor) - 1.0) * tail_mask.to(prototypes.dtype)
    alpha_vec = float(alpha) * recal_factor

    prototypes[valid_mask] = (
        (1.0 - alpha_vec[valid_mask]).unsqueeze(1) * prototypes[valid_mask]
        + alpha_vec[valid_mask].unsqueeze(1) * batch_class_mean[valid_mask]
    )
    return prototypes

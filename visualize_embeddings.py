import argparse
import os
import re

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _balanced_subsample(features, labels, max_points=4000, seed=42):
    features = _to_numpy(features)
    labels = _to_numpy(labels).astype(np.int64)

    if features.shape[0] <= max_points:
        return features, labels

    rng = np.random.default_rng(seed)
    unique_labels = np.unique(labels)
    per_class_cap = max(1, int(np.ceil(max_points / max(1, len(unique_labels)))))

    keep_indices = []
    for class_id in unique_labels:
        idx = np.flatnonzero(labels == class_id)
        if idx.size <= per_class_cap:
            keep_indices.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=per_class_cap, replace=False)
            keep_indices.extend(chosen.tolist())

    keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
    if keep_indices.size > max_points:
        keep_indices = rng.choice(keep_indices, size=max_points, replace=False)
        keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
    return features[keep_indices], labels[keep_indices]


def _compute_tsne_embedding(features, seed=42, small_n_perplexity=5):
    from sklearn.manifold import TSNE

    features = _to_numpy(features)
    if features.shape[0] < 2:
        raise ValueError("Need at least two points to compute a 2D embedding.")
    if features.shape[0] == 2:
        return np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    perplexity = max(2, min(small_n_perplexity, features.shape[0] - 1))
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)


def _compute_umap_embedding(features, seed=42):
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("UMAP visualization requires the `umap-learn` package.") from exc

    features = _to_numpy(features)
    if features.shape[0] < 2:
        raise ValueError("Need at least two points to compute a 2D embedding.")
    if features.shape[0] == 2:
        return np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    return UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=min(15, max(2, features.shape[0] - 1)),
        min_dist=0.1,
        metric="euclidean",
    ).fit_transform(features)


def _sanitize_name_for_path(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "unknown"


def save_tsne_plot(features, labels, class_names, out_path, title="", max_points=4000, seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    features, labels = _balanced_subsample(features, labels, max_points=max_points, seed=seed)
    if features.shape[0] < 2:
        return False

    embedding = _compute_tsne_embedding(features, seed=seed, small_n_perplexity=30)

    class_names = list(class_names)
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    for color_idx, class_id in enumerate(unique_labels):
        idx = labels == class_id
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=11,
            alpha=0.75,
            color=cmap(color_idx),
            label=class_name,
            linewidths=0,
        )
    ax.set_title(title or "t-SNE")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_named_point_tsne(features, class_names, out_path, title="Prototype t-SNE", seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    features = _to_numpy(features)
    if features.shape[0] < 2:
        return False

    embedding = _compute_tsne_embedding(features, seed=seed, small_n_perplexity=5)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    for class_id, class_name in enumerate(class_names):
        ax.scatter(
            embedding[class_id, 0],
            embedding[class_id, 1],
            s=120,
            alpha=0.9,
            color=cmap(class_id),
            edgecolors="black",
            linewidths=0.5,
        )
        ax.text(
            embedding[class_id, 0],
            embedding[class_id, 1],
            f" {class_name}",
            fontsize=9,
            va="center",
            ha="left",
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_prototype_mean_tsne(prototypes, means, class_names, out_path, title="Prototype vs Class Mean t-SNE", seed=42):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    prototypes = _to_numpy(prototypes)
    means = _to_numpy(means)
    if prototypes.shape != means.shape or prototypes.shape[0] < 2:
        return False

    all_points = np.concatenate([means, prototypes], axis=0)
    embedding = _compute_tsne_embedding(all_points, seed=seed, small_n_perplexity=8)
    mean_points = embedding[: len(class_names)]
    proto_points = embedding[len(class_names) :]
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    for class_id, class_name in enumerate(class_names):
        color = cmap(class_id)
        ax.plot(
            [mean_points[class_id, 0], proto_points[class_id, 0]],
            [mean_points[class_id, 1], proto_points[class_id, 1]],
            color=color,
            alpha=0.5,
            linewidth=1.2,
        )
        ax.scatter(
            mean_points[class_id, 0],
            mean_points[class_id, 1],
            s=80,
            alpha=0.85,
            color=color,
            marker="o",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.scatter(
            proto_points[class_id, 0],
            proto_points[class_id, 1],
            s=95,
            alpha=0.95,
            color=color,
            marker="^",
            edgecolors="black",
            linewidths=0.4,
        )
        ax.text(
            proto_points[class_id, 0],
            proto_points[class_id, 1],
            f" {class_name}",
            fontsize=8.5,
            va="center",
            ha="left",
        )

    ax.scatter([], [], color="gray", marker="o", label="class mean")
    ax.scatter([], [], color="gray", marker="^", label="prototype")
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_similarity_heatmap(matrix, class_names, out_path, title="Cosine Similarity Heatmap", vmin=-1.0, vmax=1.0):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    matrix = _to_numpy(matrix)
    class_names = list(class_names)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_real_virtual_tsne(
    real_features,
    real_labels,
    virtual_features,
    virtual_labels,
    class_names,
    out_path,
    title="Real vs Virtual t-SNE",
    max_real_points=2500,
    max_virtual_points=2500,
    seed=42,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    real_features, real_labels = _balanced_subsample(real_features, real_labels, max_points=max_real_points, seed=seed)
    virtual_features, virtual_labels = _balanced_subsample(
        virtual_features, virtual_labels, max_points=max_virtual_points, seed=seed + 1
    )
    if real_features.shape[0] + virtual_features.shape[0] < 2:
        return False

    features = np.concatenate([real_features, virtual_features], axis=0)
    labels = np.concatenate([real_labels, virtual_labels], axis=0)
    sources = np.concatenate(
        [
            np.zeros(real_features.shape[0], dtype=np.int64),
            np.ones(virtual_features.shape[0], dtype=np.int64),
        ],
        axis=0,
    )

    embedding = _compute_tsne_embedding(features, seed=seed, small_n_perplexity=30)

    class_names = list(class_names)
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for color_idx, class_id in enumerate(unique_labels):
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
        real_idx = (labels == class_id) & (sources == 0)
        virt_idx = (labels == class_id) & (sources == 1)
        if np.any(real_idx):
            ax.scatter(
                embedding[real_idx, 0],
                embedding[real_idx, 1],
                s=10,
                alpha=0.55,
                color=cmap(color_idx),
                marker="o",
                linewidths=0,
            )
        if np.any(virt_idx):
            ax.scatter(
                embedding[virt_idx, 0],
                embedding[virt_idx, 1],
                s=18,
                alpha=0.85,
                color=cmap(color_idx),
                marker="^",
                linewidths=0,
                label=class_name,
            )
        elif np.any(real_idx):
            ax.scatter([], [], color=cmap(color_idx), marker="o", label=class_name)

    ax.scatter([], [], color="black", marker="o", label="real")
    ax.scatter([], [], color="black", marker="^", label="virtual")
    ax.set_title(title)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_split_umap_plot(
    features,
    labels,
    splits,
    class_names,
    out_path,
    csv_path,
    title="Train / Val / Test UMAP",
    seed=42,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    features = _to_numpy(features)
    labels = _to_numpy(labels).astype(np.int64)
    splits = np.asarray(splits)

    if features.shape[0] < 2:
        return False

    embedding = _compute_umap_embedding(features, seed=seed)

    marker_map = {
        "train": "o",
        "val": "^",
        "test": "s",
    }
    split_order = ["train", "val", "test"]
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names), len(unique_labels)))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for color_idx, class_id in enumerate(unique_labels):
        class_mask = labels == class_id
        color = cmap(color_idx)
        for split_name in split_order:
            split_mask = splits == split_name
            mask = class_mask & split_mask
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=10,
                alpha=0.72,
                color=color,
                marker=marker_map[split_name],
                linewidths=0,
            )

    marker_handles = []
    for split_name in split_order:
        marker_handles.append(
            ax.scatter([], [], color="black", marker=marker_map[split_name], s=42, label=split_name)
        )
    ax.legend(handles=marker_handles, title="split", loc="best", frameon=False)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    rows = {
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "label": labels,
        "split": splits,
    }
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return True


def save_split_umap_decision_plot(
    features,
    labels,
    splits,
    predictions,
    class_names,
    out_path,
    title="Train / Val / Test UMAP (approx decision regions)",
    seed=42,
):
    try:
        import matplotlib.pyplot as plt
        from sklearn.neighbors import KNeighborsClassifier
    except ImportError:
        return False

    features = _to_numpy(features)
    labels = _to_numpy(labels).astype(np.int64)
    predictions = _to_numpy(predictions).astype(np.int64)
    splits = np.asarray(splits)

    if features.shape[0] < 2:
        return False

    embedding = _compute_umap_embedding(features, seed=seed)

    k = min(15, max(1, embedding.shape[0] - 1))
    surrogate = KNeighborsClassifier(n_neighbors=k, weights="distance")
    surrogate.fit(embedding, predictions)

    x_min, x_max = embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5
    y_min, y_max = embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 260),
        np.linspace(y_min, y_max, 260),
    )
    zz = surrogate.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    marker_map = {"train": "o", "val": "^", "test": "s"}
    split_order = ["train", "val", "test"]
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab20", max(20, len(class_names), len(unique_labels)))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.contourf(xx, yy, zz, levels=np.arange(zz.max() + 2) - 0.5, cmap=cmap, alpha=0.12)

    for color_idx, class_id in enumerate(unique_labels):
        class_mask = labels == class_id
        color = cmap(color_idx)
        for split_name in split_order:
            split_mask = splits == split_name
            mask = class_mask & split_mask
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=10,
                alpha=0.72,
                color=color,
                marker=marker_map[split_name],
                linewidths=0,
            )

    wrong_mask = predictions != labels
    if np.any(wrong_mask):
        ax.scatter(
            embedding[wrong_mask, 0],
            embedding[wrong_mask, 1],
            s=38,
            facecolors="none",
            edgecolors="red",
            linewidths=0.9,
            marker="o",
            label="wrong",
        )

    marker_handles = [
        ax.scatter([], [], color="black", marker=marker_map[split_name], s=42, label=split_name)
        for split_name in split_order
    ]
    if np.any(wrong_mask):
        marker_handles.append(ax.scatter([], [], facecolors="none", edgecolors="red", s=42, linewidths=0.9, marker="o", label="wrong"))
    ax.legend(handles=marker_handles, title="split", loc="best", frameon=False)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def save_per_class_split_umap_plots(
    features,
    labels,
    splits,
    class_names,
    out_dir,
    tag_suffix="",
    title_prefix="Per-class UMAP",
    seed=42,
    predictions=None,
):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return []

    features = _to_numpy(features)
    labels = _to_numpy(labels).astype(np.int64)
    splits = np.asarray(splits)
    predictions = None if predictions is None else _to_numpy(predictions).astype(np.int64)
    class_names = list(class_names)
    os.makedirs(out_dir, exist_ok=True)

    split_order = ["train", "val", "test"]
    marker_map = {"train": "o", "val": "^", "test": "s"}
    color_map = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    saved = []

    for class_id in np.unique(labels):
        class_mask = labels == int(class_id)
        class_features = features[class_mask]
        class_splits = splits[class_mask]
        class_predictions = None if predictions is None else predictions[class_mask]
        if class_features.shape[0] < 2:
            continue

        try:
            embedding = _compute_umap_embedding(class_features, seed=seed)
        except ValueError:
            continue

        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
        class_slug = _sanitize_name_for_path(class_name)
        stem = f"umap_class_{class_slug}"
        if tag_suffix:
            stem = f"{stem}_{tag_suffix}"
        out_path = os.path.join(out_dir, f"{stem}.png")
        csv_path = os.path.join(out_dir, f"{stem}.csv")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        marker_handles = []
        for split_name in split_order:
            split_mask = class_splits == split_name
            if not np.any(split_mask):
                continue
            ax.scatter(
                embedding[split_mask, 0],
                embedding[split_mask, 1],
                s=12,
                alpha=0.78,
                color=color_map[split_name],
                marker=marker_map[split_name],
                linewidths=0,
            )
            marker_handles.append(
                ax.scatter([], [], color=color_map[split_name], marker=marker_map[split_name], s=42, label=split_name)
            )

        if class_predictions is not None:
            wrong_mask = class_predictions != int(class_id)
            if np.any(wrong_mask):
                ax.scatter(
                    embedding[wrong_mask, 0],
                    embedding[wrong_mask, 1],
                    s=42,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=0.95,
                    marker="o",
                )
                marker_handles.append(
                    ax.scatter([], [], facecolors="none", edgecolors="red", s=42, linewidths=0.95, marker="o", label="wrong")
                )

        if marker_handles:
            ax.legend(handles=marker_handles, title="split", loc="best", frameon=False)
        ax.set_title(f"{title_prefix}: {class_name}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame(
            {
                "x": embedding[:, 0],
                "y": embedding[:, 1],
                "label": np.full(class_features.shape[0], int(class_id), dtype=np.int64),
                "split": class_splits,
                "pred": np.full(class_features.shape[0], -1, dtype=np.int64) if class_predictions is None else class_predictions,
                "correct": np.full(class_features.shape[0], True, dtype=bool) if class_predictions is None else (class_predictions == int(class_id)),
            }
        ).to_csv(csv_path, index=False)
        saved.append((class_name, out_path, csv_path))

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--class_names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="t-SNE")
    parser.add_argument("--max_points", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = torch.load(args.features, map_location="cpu")
    labels = torch.load(args.labels, map_location="cpu")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ok = save_tsne_plot(
        features=features,
        labels=labels,
        class_names=args.class_names,
        out_path=args.output,
        title=args.title,
        max_points=args.max_points,
        seed=args.seed,
    )
    if ok:
        print(f"saved to: {args.output}")
    else:
        print("t-SNE generation skipped")


if __name__ == "__main__":
    main()

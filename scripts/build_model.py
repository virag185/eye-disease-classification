from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "dataset" / "dataset"
OUT_FILE = ROOT / "app" / "assets" / "model.js"
IMAGE_SIZE = 96
GRID = 8
SEED = 2026
PROTOTYPES_PER_CLASS = 32
MAX_EXEMPLARS_PER_CLASS = 900
K_NEIGHBORS = 9


CLASS_COPY = {
    "cataract": {
        "display": "Cataract pattern",
        "accent": "#f5b84b",
        "summary": "The scan has the stronger opacity and muted-detail signature seen in the cataract examples.",
        "care": "Hackathon demo only. Confirm with an ophthalmologist before making care decisions.",
    },
    "diabetic_retinopathy": {
        "display": "Diabetic retinopathy pattern",
        "accent": "#e85d75",
        "summary": "The scan is closest to the diabetic retinopathy texture and color distribution in this dataset.",
        "care": "This is a screening-style prototype, not a diagnosis. Medical review is still required.",
    },
    "glaucoma": {
        "display": "Glaucoma pattern",
        "accent": "#56b6c2",
        "summary": "The scan aligns most with the glaucoma examples based on optic-region color and texture cues.",
        "care": "Use this as a demo signal only. Clinical glaucoma screening needs proper eye-pressure and optic-nerve assessment.",
    },
    "normal": {
        "display": "Normal pattern",
        "accent": "#67c587",
        "summary": "The scan is most similar to the normal-eye examples in the local training set.",
        "care": "A normal prototype result does not rule out disease. Seek care for symptoms or risk factors.",
    },
}


def rgb_to_hsv(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    h = np.zeros_like(mx)

    nonzero = diff > 1e-8
    red = (mx == r) & nonzero
    green = (mx == g) & nonzero
    blue = (mx == b) & nonzero

    h[red] = ((g[red] - b[red]) / diff[red]) % 6.0
    h[green] = ((b[green] - r[green]) / diff[green]) + 2.0
    h[blue] = ((r[blue] - g[blue]) / diff[blue]) + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-8, diff / mx, 0.0)
    return h, s, mx


def hist(values: np.ndarray, bins: int, weights: np.ndarray | None = None) -> list[float]:
    if values.size == 0:
        return [0.0] * bins
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), weights=weights)
    total = float(counts.sum())
    if total <= 1e-9:
        return [0.0] * bins
    return (counts / total).astype(np.float32).tolist()


def stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 5
    return [
        float(values.mean()),
        float(values.std()),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 90)),
    ]


def feature_from_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    r = arr[:, :, 0].reshape(-1)
    g = arr[:, :, 1].reshape(-1)
    b = arr[:, :, 2].reshape(-1)
    gray_2d = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
    gray = gray_2d.reshape(-1)
    mask = ((np.maximum.reduce([r, g, b]) > 0.07) & (gray > 0.03))
    if int(mask.sum()) < 500:
        mask = np.ones_like(gray, dtype=bool)

    rr, gg, bb, yy = r[mask], g[mask], b[mask], gray[mask]
    h, s, v = rgb_to_hsv(rr, gg, bb)

    features: list[float] = []
    for channel in (rr, gg, bb):
        features.extend(hist(channel, 12))
    features.extend(hist(h, 12, weights=np.maximum(s, 0.05)))
    features.extend(hist(s, 8))
    features.extend(hist(v, 8))
    features.extend(hist(yy, 16))
    for channel in (rr, gg, bb, h, s, v, yy):
        features.extend(stats(channel))

    yy_grid = gray_2d
    mask_2d = mask.reshape(IMAGE_SIZE, IMAGE_SIZE)
    step = IMAGE_SIZE // GRID
    for gy in range(GRID):
        for gx in range(GRID):
            cell = yy_grid[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
            cell_mask = mask_2d[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
            features.append(float(cell[cell_mask].mean()) if cell_mask.any() else 0.0)

    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    dist = np.sqrt((x - IMAGE_SIZE / 2) ** 2 + (y - IMAGE_SIZE / 2) ** 2) / (IMAGE_SIZE / 2)
    center = (dist < 0.33) & mask_2d
    ring = (dist >= 0.33) & (dist < 0.72) & mask_2d
    outer = (dist >= 0.72) & mask_2d
    for region in (center, ring, outer):
        features.append(float(gray_2d[region].mean()) if region.any() else 0.0)
    center_mean = features[-3]
    outer_mean = features[-1]
    features.append(center_mean - outer_mean)

    dx = np.abs(gray_2d[:, 1:] - gray_2d[:, :-1])
    dy = np.abs(gray_2d[1:, :] - gray_2d[:-1, :])
    features.extend([float(dx.mean()), float(dx.std()), float(dy.mean()), float(dy.std())])
    return np.asarray(features, dtype=np.float32)


def collect_dataset() -> tuple[list[str], list[Path], list[str]]:
    classes = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])
    paths: list[Path] = []
    labels: list[str] = []
    for cls in classes:
        files = sorted((DATASET_DIR / cls).glob("*.*"))
        paths.extend(files)
        labels.extend([cls] * len(files))
    return classes, paths, labels


def stratified_split(labels: list[str], holdout: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(SEED)
    by_class: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(label, []).append(index)
    train, val = [], []
    for items in by_class.values():
        rng.shuffle(items)
        cut = max(1, int(len(items) * holdout))
        val.extend(items[:cut])
        train.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return np.asarray(train), np.asarray(val)


def kmeans(points: np.ndarray, k: int, iterations: int = 28) -> np.ndarray:
    rng = np.random.default_rng(SEED + points.shape[0] + k)
    if len(points) <= k:
        return points.copy()
    centers = points[rng.choice(len(points), size=k, replace=False)].copy()
    for _ in range(iterations):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        next_centers = centers.copy()
        for i in range(k):
            members = points[assignment == i]
            if len(members):
                next_centers[i] = members.mean(axis=0)
            else:
                next_centers[i] = points[rng.integers(0, len(points))]
        if np.allclose(centers, next_centers, atol=1e-4):
            break
        centers = next_centers
    return centers.astype(np.float32)


def build_prototypes(features: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict[str, np.ndarray]:
    return {
        cls: kmeans(features[labels == cls], PROTOTYPES_PER_CLASS)
        for cls in classes
    }


def predict(features: np.ndarray, prototypes: dict[str, np.ndarray], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    all_scores = []
    for cls in classes:
        dists = ((features[:, None, :] - prototypes[cls][None, :, :]) ** 2).sum(axis=2)
        nearest = dists.min(axis=1)
        all_scores.append(nearest)
    distances = np.stack(all_scores, axis=1)
    logits = -distances
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs, distances


def predict_knn(features: np.ndarray, exemplars: np.ndarray, exemplar_labels: np.ndarray, classes: list[str]) -> np.ndarray:
    probs = []
    label_to_index = {label: index for index, label in enumerate(classes)}
    for row in features:
        distances = ((exemplars - row) ** 2).sum(axis=1)
        nearest = np.argpartition(distances, K_NEIGHBORS)[:K_NEIGHBORS]
        weights = 1.0 / (distances[nearest] + 1e-4)
        scores = np.zeros(len(classes), dtype=np.float32)
        for idx, weight in zip(nearest, weights):
            scores[label_to_index[exemplar_labels[idx]]] += float(weight)
        scores = scores / max(float(scores.sum()), 1e-8)
        probs.append(scores)
    return np.vstack(probs)


def rounded(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, np.ndarray):
        return rounded(obj.tolist())
    if isinstance(obj, list):
        return [rounded(x) for x in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return obj


def main() -> None:
    classes, paths, labels = collect_dataset()
    features = []
    for index, path in enumerate(paths, start=1):
        features.append(feature_from_image(path))
        if index % 500 == 0:
            print(f"extracted {index}/{len(paths)} images")
    x = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)

    train_idx, val_idx = stratified_split(labels)
    train_x = x[train_idx]
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-5] = 1.0
    z_train = (train_x - mean) / std
    z_val = (x[val_idx] - mean) / std
    split_prototypes = build_prototypes(z_train, y[train_idx], classes)
    probs, _ = predict(z_val, split_prototypes, classes)
    pred = np.asarray(classes, dtype=object)[probs.argmax(axis=1)]
    actual = y[val_idx]
    prototype_accuracy = float((pred == actual).mean())

    knn_probs = predict_knn(z_val, z_train, y[train_idx], classes)
    knn_pred = np.asarray(classes, dtype=object)[knn_probs.argmax(axis=1)]
    accuracy = float((knn_pred == actual).mean())
    confusion = {
        cls: {inner: int(((actual == cls) & (knn_pred == inner)).sum()) for inner in classes}
        for cls in classes
    }

    all_mean = x.mean(axis=0)
    all_std = x.std(axis=0)
    all_std[all_std < 1e-5] = 1.0
    z_all = (x - all_mean) / all_std
    final_prototypes = build_prototypes(z_all, y, classes)
    exemplar_indices = []
    rng = np.random.default_rng(SEED)
    for cls in classes:
        class_indices = np.where(y == cls)[0]
        if len(class_indices) > MAX_EXEMPLARS_PER_CLASS:
            class_indices = rng.choice(class_indices, size=MAX_EXEMPLARS_PER_CLASS, replace=False)
        exemplar_indices.extend(class_indices.tolist())
    exemplar_indices = sorted(exemplar_indices)
    exemplar_features = z_all[exemplar_indices]
    exemplar_labels = y[exemplar_indices].tolist()

    samples = {}
    for cls in classes:
        files = sorted((DATASET_DIR / cls).glob("*.*"))[:6]
        samples[cls] = [
            str(path.relative_to(ROOT)).replace("\\", "/")
            for path in files
        ]

    model = {
        "version": "retinascan-lite-1",
        "imageSize": IMAGE_SIZE,
        "grid": GRID,
        "classes": classes,
        "classInfo": {cls: CLASS_COPY.get(cls, {}) for cls in classes},
        "counts": {cls: int((y == cls).sum()) for cls in classes},
        "validation": {
            "accuracy": accuracy,
            "prototypeAccuracy": prototype_accuracy,
            "samples": int(len(val_idx)),
            "confusion": confusion,
            "seed": SEED,
            "note": "Validation uses handcrafted image features plus weighted nearest neighbors; it is a hackathon prototype, not a clinical model.",
        },
        "kNeighbors": K_NEIGHBORS,
        "featureMean": all_mean,
        "featureStd": all_std,
        "prototypes": final_prototypes,
        "exemplars": {
            "labels": exemplar_labels,
            "features": exemplar_features,
        },
        "samples": samples,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("window.RETINASCAN_MODEL = " + json.dumps(rounded(model), separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT_FILE}")
    print(f"validation accuracy: {accuracy:.3f} on {len(val_idx)} images")
    print(f"prototype-only accuracy: {prototype_accuracy:.3f}")


if __name__ == "__main__":
    main()

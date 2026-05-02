from __future__ import annotations

import cgi
import io
import json
import math
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "app" / "assets" / "model.js"
DATASET_ROOT = ROOT / "dataset" / "dataset"


def load_model() -> dict:
    text = MODEL_FILE.read_text(encoding="utf-8").strip()
    prefix = "window.RETINASCAN_MODEL = "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(";"):
        text = text[:-1]
    return json.loads(text)


MODEL = load_model()


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
    return (counts / total).astype(np.float32).tolist() if total > 1e-9 else [0.0] * bins


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


def extract_features(image: Image.Image) -> np.ndarray:
    size = int(MODEL["imageSize"])
    grid = int(MODEL["grid"])
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    r = arr[:, :, 0].reshape(-1)
    g = arr[:, :, 1].reshape(-1)
    b = arr[:, :, 2].reshape(-1)
    gray_2d = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gray = gray_2d.reshape(-1)
    mask = (np.maximum.reduce([r, g, b]) > 0.07) & (gray > 0.03)
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

    step = size // grid
    mask_2d = mask.reshape(size, size)
    for gy in range(grid):
        for gx in range(grid):
            cell = gray_2d[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
            cell_mask = mask_2d[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
            features.append(float(cell[cell_mask].mean()) if cell_mask.any() else 0.0)

    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2) / (size / 2)
    center = (dist < 0.33) & mask_2d
    ring = (dist >= 0.33) & (dist < 0.72) & mask_2d
    outer = (dist >= 0.72) & mask_2d
    for region in (center, ring, outer):
        features.append(float(gray_2d[region].mean()) if region.any() else 0.0)
    features.append(features[-3] - features[-1])

    dx = np.abs(gray_2d[:, 1:] - gray_2d[:, :-1])
    dy = np.abs(gray_2d[1:, :] - gray_2d[:-1, :])
    features.extend([float(dx.mean()), float(dx.std()), float(dy.mean()), float(dy.std())])
    return np.asarray(features, dtype=np.float32)


def image_quality(image: Image.Image) -> dict:
    size = 128
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lap = np.abs(4 * gray[1:-1, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:] - gray[:-2, 1:-1] - gray[2:, 1:-1])
    return {
        "brightness": round(float(np.clip(gray.mean() * 100, 0, 100)), 2),
        "contrast": round(float(np.clip(gray.std() * 260, 0, 100)), 2),
        "sharpness": round(float(np.clip(lap.mean() * 950, 0, 100)), 2),
    }


def classify(image: Image.Image) -> dict:
    raw = extract_features(image)
    mean = np.asarray(MODEL["featureMean"], dtype=np.float32)
    std = np.asarray(MODEL["featureStd"], dtype=np.float32)
    normalized = (raw - mean) / std
    labels = MODEL["exemplars"]["labels"]
    exemplars = np.asarray(MODEL["exemplars"]["features"], dtype=np.float32)
    distances = ((exemplars - normalized) ** 2).sum(axis=1)
    k = int(MODEL.get("kNeighbors", 9))
    nearest = np.argpartition(distances, k)[:k]
    scores = {cls: 0.0 for cls in MODEL["classes"]}
    for idx in nearest:
        scores[labels[int(idx)]] += 1.0 / (float(distances[int(idx)]) + 0.0001)
    total = sum(scores.values()) or 1.0
    ranks = [
        {
            "cls": cls,
            "probability": scores[cls] / total,
            "label": MODEL["classInfo"].get(cls, {}).get("display", cls).replace(" pattern", ""),
        }
        for cls in MODEL["classes"]
    ]
    ranks.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "ranks": ranks,
        "quality": image_quality(image),
        "model": {
            "version": MODEL["version"],
            "validationAccuracy": MODEL["validation"]["accuracy"],
        },
    }


def safe_sample_path(raw_path: str) -> Path:
    normalized = unquote(raw_path).replace("\\", "/")
    if normalized.startswith("dataset/dataset/"):
        normalized = normalized[len("dataset/dataset/") :]
    candidate = (DATASET_ROOT / normalized).resolve()
    if DATASET_ROOT.resolve() not in candidate.parents:
        raise ValueError("Sample path is outside the dataset directory.")
    if not candidate.exists():
        raise FileNotFoundError(str(candidate))
    return candidate


class RetinaForgeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json({"ok": True, "model": MODEL["version"]})
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            if self.path == "/api/analyze":
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                field = form["image"] if "image" in form else None
                if field is None or not getattr(field, "file", None):
                    self.send_json({"error": "No image file provided."}, 400)
                    return
                data = field.file.read()
                image = Image.open(io.BytesIO(data))
                self.send_json(classify(image))
                return

            if self.path == "/api/analyze-sample":
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                sample_path = safe_sample_path(payload.get("path", ""))
                image = Image.open(sample_path)
                self.send_json(classify(image))
                return

            self.send_json({"error": "Unknown endpoint."}, 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, 500)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8080), RetinaForgeHandler)
    print("RetinaForge backend running at http://127.0.0.1:8080/app/")
    server.serve_forever()


if __name__ == "__main__":
    main()

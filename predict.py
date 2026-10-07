"""
predict.py: Inference for the handwritten character classifier.

Usage:
  python predict.py path/to/image.png
  python predict.py path/to/folder --top-k 5
  python predict.py img.png --model outputs/best_model.keras --labels outputs/label_classes.npy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorflow import keras

from train import preprocess_image  # single source of truth for preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("predict")

DEFAULT_DIR = Path(r"C:\Users\sivag\OneDrive\Desktop\ALPHA\PRO1\outputs")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class CharacterPredictor:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        self.model = keras.models.load_model(model_path)
        self.classes = np.load(labels_path, allow_pickle=False)
        self.size = int(self.model.input_shape[1])  # derive size from the model, not a constant

        n_out = int(self.model.output_shape[-1])
        if n_out != len(self.classes):
            raise ValueError(f"Model has {n_out} outputs but labels file has {len(self.classes)} classes. "
                             "Model and labels must come from the same training run.")
        log.info("Loaded model (input %dx%d, %d classes)", self.size, self.size, n_out)

    def _load_batch(self, paths: List[Path]) -> Tuple[np.ndarray, List[Path]]:
        imgs, ok = [], []
        for p in paths:
            img = preprocess_image(p, self.size)
            if img is None:
                log.warning("Skipping unreadable image: %s", p)
                continue
            imgs.append(img)
            ok.append(p)
        if not imgs:
            return np.empty((0, self.size, self.size, 1), dtype="float32"), ok
        return np.stack(imgs)[..., None].astype("float32"), ok  # raw 0-255, model rescales internally

    def predict(self, paths: List[Path], top_k: int = 3, batch_size: int = 64):
        """Return [(path, [(label, prob), ...top_k]), ...]."""
        top_k = min(top_k, len(self.classes))
        results = []
        for i in range(0, len(paths), batch_size):
            X, ok = self._load_batch(paths[i:i + batch_size])
            if len(ok) == 0:
                continue
            probs = self.model.predict(X, verbose=0)
            # argpartition: O(C) per row, then sort only the k winners
            idx = np.argpartition(-probs, top_k - 1, axis=1)[:, :top_k]
            for row, (p, ids) in enumerate(zip(ok, idx)):
                ids = ids[np.argsort(-probs[row, ids])]
                results.append((p, [(str(self.classes[j]), float(probs[row, j])) for j in ids]))
        return results


def collect_paths(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"Not a file or folder: {target}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Image file or folder of images")
    ap.add_argument("--model", type=Path, default=DEFAULT_DIR / "best_model.keras")
    ap.add_argument("--labels", type=Path, default=DEFAULT_DIR / "label_classes.npy")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--low-conf", type=float, default=0.5,
                    help="Flag predictions below this confidence")
    args = ap.parse_args()

    try:
        paths = collect_paths(args.input)
        if not paths:
            log.error("No images found in %s", args.input)
            return 1
        predictor = CharacterPredictor(args.model, args.labels)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1

    print(f"\n{'FILE':<40} {'PRED':<6} {'CONF':>7}   TOP-{args.top_k}")
    print("-" * 80)
    for path, top in predictor.predict(paths, args.top_k):
        label, conf = top[0]
        flag = "  (low confidence)" if conf < args.low_conf else ""
        alts = ", ".join(f"{l}:{p:.0%}" for l, p in top)
        print(f"{path.name[:39]:<40} {label:<6} {conf:>6.1%}   {alts}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
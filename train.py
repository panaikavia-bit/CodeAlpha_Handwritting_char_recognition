from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train")

# Letters whose lower/upper forms are visually near-identical.
AMBIGUOUS_CASE = set("ckmopsuvwxyz")


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _real_csv_candidates(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith(".")
    )


DATA_DIR_CANDIDATES = (
    PROJECT_ROOT / "DATA" / "img",
    PROJECT_ROOT / "DATA" / "Img",
    PROJECT_ROOT / "DATA",
)
CSV_CANDIDATES = (
    PROJECT_ROOT / "DATA" / "english.csv",
    PROJECT_ROOT / "DATA" / "English.csv",
    PROJECT_ROOT / "DATA" / "english.CSV",
    PROJECT_ROOT / "DATA" / "labels.csv",
    PROJECT_ROOT / "DATA" / "data.csv",
)


@dataclass
class Config:
    data_dir: Path = _first_existing(*DATA_DIR_CANDIDATES)
    csv_path: Path = _first_existing(*CSV_CANDIDATES)
    out_dir: Path = PROJECT_ROOT / "outputs"
    model: str = "mobilenet"          # "mobilenet" | "cnn"
    img_size: int = 96                # MobileNetV2 supports 96/128/160/192/224
    batch_size: int = 32
    seed: int = 42
    epochs_head: int = 25             # mobilenet phase 1, or total epochs for cnn
    epochs_finetune: int = 40         # mobilenet phase 2
    finetune_layers: int = 50
    label_smoothing: float = 0.1
    merge_case: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----------------------------------------------------------------------
# Preprocessing: O(H*W) per image; run once, then cached
# ----------------------------------------------------------------------
def preprocess_image(path: Path, size: int) -> Optional[np.ndarray]:
    """Return a (size, size) uint8 image: dark ink on a light background, centered."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 1. Polarity: the border median approximates the background. Force a light background.
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if np.median(border) < 127:
        img = 255 - img

    # 2. Contrast stretch
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Crop to the ink bounding box (Otsu on inverted image). Fallback: keep the full image.
    _, mask = cv2.threshold(255 - img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) > 50:
        h, w = img.shape
        pad_y = int(0.05 * h)
        pad_x = int(0.05 * w)
        y0, y1 = max(ys.min() - pad_y, 0), min(ys.max() + pad_y + 1, h)
        x0, x1 = max(xs.min() - pad_x, 0), min(xs.max() + pad_x + 1, w)
        if (y1 - y0) > 8 and (x1 - x0) > 8:
            img = img[y0:y1, x0:x1]

    # 4. Letterbox to a square with a 10% margin (keeps the aspect ratio)
    h, w = img.shape
    side = int(max(h, w) * 1.1)
    canvas = np.full((side, side), 255, dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = img

    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def load_dataset(cfg: Config) -> Tuple[np.ndarray, List[str]]:
    cache = cfg.out_dir / f"cache_{cfg.img_size}.npz"
    if cache.exists():
        log.info("Loading cached dataset: %s", cache)
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["y"].tolist()

    data_dir = cfg.data_dir if cfg.data_dir.exists() else PROJECT_ROOT / "DATA"
    real_csvs = _real_csv_candidates(data_dir)
    if not cfg.csv_path.exists() or cfg.csv_path.name.startswith("."):
        if real_csvs:
            cfg.csv_path = real_csvs[0]
        else:
            raise FileNotFoundError(
                f"Dataset CSV not found. Looked for: {cfg.csv_path}. "
                f"The DATA directory has no real CSV labels file. "
                f"Check that the exported dataset includes a non-hidden CSV such as english.csv."
            )

    raw = cfg.csv_path.read_bytes()[:200]
    if b"Mac OS X" in raw or b"com.apple.quarantine" in raw:
        raise ValueError(
            f"{cfg.csv_path} is not the dataset labels file; it appears to be a macOS metadata/quarantine file. "
            "Remove the hidden metadata file and keep only the real CSV exported from the dataset."
        )

    df = pd.read_csv(cfg.csv_path)
    if not {"image", "label"}.issubset(df.columns):
        raise ValueError(f"CSV must have 'image' and 'label' columns, got {list(df.columns)}")

    images, labels, skipped = [], [], 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        path = cfg.data_dir / os.path.basename(row.image)
        img = preprocess_image(path, cfg.img_size)
        if img is None:
            skipped += 1
            log.warning("Unreadable image skipped: %s", path)
            continue
        images.append(img)
        labels.append(str(row.label))
        if i % 500 == 0:
            log.info("Processed %d/%d", i, len(df))

    if not images:
        raise RuntimeError("No images loaded. Check DATA paths.")
    X = np.stack(images)[..., None]  # (N, S, S, 1) uint8
    log.info("Loaded %d images (%d skipped)", len(X), skipped)
    np.savez_compressed(cache, X=X, y=np.array(labels))
    return X, labels


# ----------------------------------------------------------------------
# tf.data
# ----------------------------------------------------------------------
def make_ds(X: np.ndarray, y: np.ndarray, num_classes: int, batch: int,
            training: bool, seed: int) -> tf.data.Dataset:
    y_oh = keras.utils.to_categorical(y, num_classes)
    ds = tf.data.Dataset.from_tensor_slices((X.astype("float32"), y_oh))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------
# Models (inputs are float32 in [0, 255]; rescaling happens inside the model)
# ----------------------------------------------------------------------
def augmentation() -> keras.Sequential:
    # Active only when training=True. No flips: they would change character identity.
    return keras.Sequential([
        layers.RandomRotation(0.06, fill_mode="nearest"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        layers.RandomZoom(0.12, fill_mode="nearest"),
        layers.RandomContrast(0.2),
    ], name="augment")


def conv_block(x, filters: int, drop: float):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    return layers.Dropout(drop)(x)


def build_cnn(size: int, num_classes: int) -> keras.Model:
    inp = layers.Input((size, size, 1))
    x = augmentation()(inp)
    x = layers.Rescaling(1.0 / 255)(x)
    for f, d in [(32, 0.1), (64, 0.15), (128, 0.2), (256, 0.25)]:
        x = conv_block(x, f, d)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="cnn_scratch")


def build_mobilenet(size: int, num_classes: int) -> Tuple[keras.Model, keras.Model]:
    base = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inp = layers.Input((size, size, 1))
    x = augmentation()(inp)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)   # MobileNetV2 expects [-1, 1]
    x = layers.Concatenate()([x, x, x])                  # gray -> 3 channels
    x = base(x, training=False)                          # keeps BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mobilenetv2_gray"), base


def compile_model(model: keras.Model, lr: float, smoothing: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"),
                 keras.metrics.TopKCategoricalAccuracy(3, name="top3")],
    )


def callbacks(ckpt: Path, patience: int) -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping("val_accuracy", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=max(patience // 3, 3),
                                          min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_accuracy", mode="max",
                                        save_best_only=True),
    ]


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    X, labels = load_dataset(cfg)
    if cfg.merge_case:
        labels = [l.upper() if l.lower() in AMBIGUOUS_CASE else l for l in labels]
        log.info("Merged ambiguous upper/lower pairs.")

    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    num_classes = len(enc.classes_)
    log.info("Classes: %d", num_classes)
    np.save(cfg.out_dir / "label_classes.npy", enc.classes_)

    # 70/15/15 stratified, via two splits
    idx = np.arange(len(X))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=cfg.seed)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp],
                                       random_state=cfg.seed)
    np.savez(cfg.out_dir / "split_indices.npz", train=idx_tr, val=idx_val, test=idx_te)
    log.info("Split: train=%d val=%d test=%d", len(idx_tr), len(idx_val), len(idx_te))

    train_ds = make_ds(X[idx_tr], y[idx_tr], num_classes, cfg.batch_size, True, cfg.seed)
    val_ds = make_ds(X[idx_val], y[idx_val], num_classes, cfg.batch_size, False, cfg.seed)
    test_ds = make_ds(X[idx_te], y[idx_te], num_classes, cfg.batch_size, False, cfg.seed)

    ckpt = cfg.out_dir / "best_model.keras"
    history = {}

    if cfg.model == "cnn":
        model = build_cnn(cfg.img_size, num_classes)
        compile_model(model, 1e-3, cfg.label_smoothing)
        model.summary()
        h = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_head + cfg.epochs_finetune,
                      callbacks=callbacks(ckpt, patience=15), verbose=2)
        history["train"] = h.history
    else:
        model, base = build_mobilenet(cfg.img_size, num_classes)

        # Phase 1: train the head only
        compile_model(model, 1e-3, cfg.label_smoothing)
        model.summary()
        log.info("Phase 1: head training")
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_head,
                       callbacks=callbacks(ckpt, patience=10), verbose=2)

        # Phase 2: unfreeze the top layers; BatchNorm layers stay frozen
        base.trainable = True
        for layer in base.layers[:-cfg.finetune_layers]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, 1e-4, cfg.label_smoothing)   # recompile after changing trainable
        log.info("Phase 2: fine-tuning top %d layers", cfg.finetune_layers)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_finetune,
                       callbacks=callbacks(ckpt, patience=12), verbose=2)
        history["head"], history["finetune"] = h1.history, h2.history

    # Final one-shot test evaluation (best weights restored by EarlyStopping)
    loss, acc, top3 = model.evaluate(test_ds, verbose=0)
    log.info("TEST accuracy: %.2f%% | top-3: %.2f%% | loss: %.4f", acc * 100, top3 * 100, loss)

    json.dump({"test_accuracy": float(acc), "test_top3": float(top3), "test_loss": float(loss),
               "model": cfg.model, "merge_case": cfg.merge_case, "num_classes": num_classes},
              open(cfg.out_dir / "test_metrics.json", "w"), indent=2)
    json.dump(history, open(cfg.out_dir / "history.json", "w"),
              default=lambda o: [float(v) for v in o])
    log.info("Artifacts saved to %s", cfg.out_dir)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mobilenet", "cnn"], default="mobilenet")
    p.add_argument("--merge-case", action="store_true")
    p.add_argument("--img-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    return Config(model=a.model, merge_case=a.merge_case, img_size=a.img_size,
                  batch_size=a.batch_size)


if __name__ == "__main__":
    train(parse_args())

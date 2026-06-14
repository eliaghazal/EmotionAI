"""FER2013 dataset loader with augmentation and class-imbalance handling.

Supports two dataset layouts:
  1. Image folders  (Kaggle ZIP format):
       archive/train/{emotion}/*.jpg
       archive/test/{emotion}/*.jpg
  2. CSV flat file  (original FER2013 format):
       data/fer2013.csv
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
# Folder names in the Kaggle archive → FER2013 class index
FOLDER_TO_CLASS: Dict[str, int] = {
    "angry":    0,
    "disgust":  1,
    "fear":     2,
    "happy":    3,
    "sad":      4,
    "surprise": 5,
    "neutral":  6,
}
EMOTION_LABELS: Dict[int, str] = {v: k.capitalize() for k, v in FOLDER_TO_CLASS.items()}
EMOTION_LABELS[6] = "Neutral"   # keep consistent capitalisation

NUM_CLASSES = 7
IMG_SIZE = 48

AUGMENTATION_CONFIG = {
    "rotation_range": 10,
    "zoom_range": 0.10,
    "horizontal_flip": True,
    "brightness_range": [0.85, 1.15],
    "fill_mode": "nearest",
}
VAL_SPLIT = 0.10   # fraction of train images held out for validation


def _load_split_from_folder(split_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read all JPEGs under split_dir/{emotion}/ into arrays.

    Args:
        split_dir: Path to a split folder (e.g. archive/train or archive/test).

    Returns:
        (X, y_int) where X is float32 (N, 48, 48, 1) in [0,1]
        and y_int is int32 class indices (N,).
    """
    images: List[np.ndarray] = []
    labels: List[int] = []

    emotion_dirs = sorted([d for d in split_dir.iterdir()
                           if d.is_dir() and d.name.lower() in FOLDER_TO_CLASS])
    if not emotion_dirs:
        raise FileNotFoundError(
            f"No emotion sub-folders found inside '{split_dir}'.\n"
            f"Expected folders: {list(FOLDER_TO_CLASS.keys())}"
        )

    for emo_dir in emotion_dirs:
        class_idx = FOLDER_TO_CLASS[emo_dir.name.lower()]
        img_files = sorted(emo_dir.glob("*.jpg")) + sorted(emo_dir.glob("*.png"))
        for img_path in tqdm(img_files, desc=f"  {emo_dir.name:10s}", leave=False):
            try:
                img = Image.open(img_path).convert("L")   # grayscale
                arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
                images.append(arr[..., np.newaxis])        # (48,48,1)
                labels.append(class_idx)
            except Exception as exc:
                logger.warning("Skipping %s: %s", img_path, exc)

    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y


def load_fer2013_from_folders(
    archive_dir: str,
    val_split: float = VAL_SPLIT,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Load FER2013 from the Kaggle image-folder layout.

    Args:
        archive_dir: Path to the 'archive' folder that contains 'train/' and 'test/'.
        val_split: Fraction of training images to hold out for validation.
        seed: Random seed for reproducible train/val split.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
        X arrays: float32 (N, 48, 48, 1) in [0, 1].
        y arrays: one-hot float32 (N, 7).
    """
    root = Path(archive_dir)
    train_dir = root / "train"
    test_dir = root / "test"

    for d in (train_dir, test_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Expected '{d}' inside archive_dir='{archive_dir}'.\n"
                "Make sure the Kaggle FER2013 archive is extracted correctly."
            )

    logger.info("Loading training images from %s …", train_dir)
    X_full, y_full = _load_split_from_folder(train_dir)

    # Reproducible shuffle then split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_full))
    val_n = int(len(X_full) * val_split)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    logger.info("Loading test images from %s …", test_dir)
    X_test, y_test_int = _load_split_from_folder(test_dir)

    y_train_oh = to_categorical(y_train, NUM_CLASSES).astype(np.float32)
    y_val_oh = to_categorical(y_val, NUM_CLASSES).astype(np.float32)
    y_test_oh = to_categorical(y_test_int, NUM_CLASSES).astype(np.float32)

    logger.info(
        "Dataset loaded — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh


def load_fer2013(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
    """Load and split the FER2013 CSV into train/val/test arrays (legacy format).

    Args:
        csv_path: Absolute path to fer2013.csv.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"FER2013 CSV not found at '{csv_path}'.\n"
            "If you have the Kaggle image-folder archive, use "
            "load_fer2013_from_folders() instead."
        )

    logger.info("Loading FER2013 CSV from %s …", csv_path)
    df = pd.read_csv(csv_path)

    def parse_pixels(pixel_str: str) -> np.ndarray:
        arr = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
        return arr.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32) / 255.0

    X_all = np.stack(df["pixels"].apply(parse_pixels).values)
    y_all = df["emotion"].values

    if "Usage" in df.columns:
        train_mask = df["Usage"] == "Training"
        pub_mask = df["Usage"] == "PublicTest"
        priv_mask = df["Usage"] == "PrivateTest"
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[pub_mask], y_all[pub_mask]
        X_test, y_test = X_all[priv_mask], y_all[priv_mask]
    else:
        n = len(X_all)
        idx = np.random.permutation(n)
        t, v = int(0.8 * n), int(0.9 * n)
        X_train, y_train = X_all[idx[:t]], y_all[idx[:t]]
        X_val, y_val = X_all[idx[t:v]], y_all[idx[t:v]]
        X_test, y_test = X_all[idx[v:]], y_all[idx[v:]]

    y_train_oh = to_categorical(y_train, NUM_CLASSES).astype(np.float32)
    y_val_oh = to_categorical(y_val, NUM_CLASSES).astype(np.float32)
    y_test_oh = to_categorical(y_test, NUM_CLASSES).astype(np.float32)

    logger.info(
        "Dataset loaded — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh


def compute_class_weights(y_train_oh: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights to counter FER2013 class imbalance.

    Args:
        y_train_oh: One-hot training labels, shape (N, 7).

    Returns:
        Dict mapping class index to sample weight scalar.
    """
    y_int = np.argmax(y_train_oh, axis=1)
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight("balanced", classes=classes, y=y_int)
    cw = {int(i): float(w) for i, w in enumerate(weights)}
    logger.info("Class weights: %s", cw)
    return cw


def build_train_generator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 64,
) -> "ImageDataGenerator":
    """Create an augmented training data generator.

    .. deprecated::
        Use :func:`build_tf_dataset` instead.  ``ImageDataGenerator.flow()``
        has a Keras 3.x epoch-boundary bug where every other epoch receives
        only one step of data.

    Args:
        X_train: Training images, shape (N, 48, 48, 1), values in [0, 1].
        y_train: One-hot labels, shape (N, 7).
        batch_size: Mini-batch size.

    Returns:
        A Keras ImageDataGenerator iterator.
    """
    gen = ImageDataGenerator(**AUGMENTATION_CONFIG)
    return gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)


def build_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    augment: bool = True,
    mobilenet_preprocess: bool = False,
    mixup_alpha: float = 0.0,
) -> "tf.data.Dataset":
    """Build a repeating ``tf.data.Dataset`` with optional augmentation.

    Unlike ``ImageDataGenerator.flow()``, this dataset calls ``.repeat()``
    so it **never exhausts mid-epoch** (fixes the Keras 3.x epoch-boundary
    bug).  Pass ``mobilenet_preprocess=True`` to scale images from ``[0, 1]``
    to ``[-1, 1]`` after augmentation (required for MobileNetV2).

    Pass ``mixup_alpha > 0`` to enable MixUp batch-level augmentation.
    MixUp blends pairs of training samples and their labels, acting as a
    strong regulariser on FER2013's noisy annotations.

    Args:
        X: Images float32 ``(N, H, W, C)`` in ``[0, 1]``.
        y: One-hot labels float32 ``(N, num_classes)``.
        batch_size: Mini-batch size.
        augment: Apply random flip / brightness / contrast / zoom.
        mobilenet_preprocess: Scale to ``[-1, 1]`` for MobileNetV2.
        mixup_alpha: If > 0, enable MixUp augmentation.  A value of 0.2
            is recommended; set to 0.0 to disable.

    Returns:
        An infinitely repeating, prefetched ``tf.data.Dataset``.
    """
    n, h, w, c = X.shape
    min_crop = int(0.9 * h)

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=min(n, 10_000), reshuffle_each_iteration=True)

    if augment:
        def _aug(img: tf.Tensor, lbl: tf.Tensor):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.15)
            img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
            # Zoom simulation: random crop then resize back to original dims
            crop_h = tf.random.uniform((), minval=min_crop, maxval=h + 1, dtype=tf.int32)
            img = tf.image.random_crop(img, size=[crop_h, crop_h, c])
            img = tf.image.resize(img, [h, w])
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, lbl

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    if mobilenet_preprocess:
        # MobileNetV2 expects inputs in [-1, 1]; our float arrays are in [0, 1]
        def _scale(img: tf.Tensor, lbl: tf.Tensor):
            return img * 2.0 - 1.0, lbl
        ds = ds.map(_scale, num_parallel_calls=tf.data.AUTOTUNE)

    # MixUp requires consistent batch sizes so the last partial batch is dropped
    ds = ds.batch(batch_size, drop_remainder=(mixup_alpha > 0.0))

    if mixup_alpha > 0.0:
        def _mixup(images: tf.Tensor, labels: tf.Tensor):
            """Blend pairs of samples within the batch (MixUp augmentation).

            lambda ~ Uniform(0.5, 1.0) so the primary sample always dominates,
            preventing label inversion on FER2013's already-noisy annotations.
            """
            batch_n = tf.shape(images)[0]
            lam = tf.random.uniform(())
            lam = tf.maximum(lam, 1.0 - lam)       # force lam in [0.5, 1.0]
            idx = tf.random.shuffle(tf.range(batch_n))
            mixed_images = lam * images + (1.0 - lam) * tf.gather(images, idx)
            mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, idx)
            return mixed_images, mixed_labels

        ds = ds.map(_mixup, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.repeat()       # CRITICAL — prevents "ran out of data" every other epoch
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def apply_mobilenet_scale(X: np.ndarray) -> np.ndarray:
    """Scale images from ``[0, 1]`` to ``[-1, 1]`` for MobileNetV2.

    Use this on validation / test arrays that are fed directly to
    ``model.evaluate()`` or ``validation_data``.

    Args:
        X: Float32 images in ``[0, 1]``.

    Returns:
        Float32 images in ``[-1, 1]``.
    """
    return (X * 2.0 - 1.0).astype(np.float32)


def preprocess_for_mobilenet(
    X: np.ndarray,
    target_size: int = 96,
) -> np.ndarray:
    """Upsample grayscale images and stack channels for MobileNetV2.

    Returns values in ``[0, 1]``.  To get the ``[-1, 1]`` range that
    MobileNetV2 expects, pass the output through :func:`apply_mobilenet_scale`
    or use ``mobilenet_preprocess=True`` in :func:`build_tf_dataset`.

    Args:
        X: Images of shape (N, 48, 48, 1) in [0, 1].
        target_size: Spatial size expected by MobileNetV2 (96).

    Returns:
        Float32 array of shape (N, target_size, target_size, 3) in [0, 1].
    """
    import cv2

    out = np.zeros((len(X), target_size, target_size, 3), dtype=np.float32)
    for i, img in enumerate(X):
        gray = (img[:, :, 0] * 255).astype(np.uint8)
        resized = cv2.resize(gray, (target_size, target_size))
        rgb = np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
        out[i] = rgb
    return out

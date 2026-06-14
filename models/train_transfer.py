"""Two-phase MobileNetV2 transfer learning on FER2013."""

# ── macOS SSL fix: point urllib at certifi's CA bundle ────────────────────────
# Python 3.13 from python.org ships without system certs wired up.
import os as _os
try:
    import certifi as _certifi
    _os.environ.setdefault("SSL_CERT_FILE", _certifi.where())
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _certifi.where())
except ImportError:
    pass  # certifi not installed — rely on system certs
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.dataset_loader import (
    apply_mobilenet_scale,
    build_tf_dataset,
    compute_class_weights,
    load_fer2013_from_folders,
    preprocess_for_mobilenet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
@dataclass
class TransferConfig:
    img_size: int = 96
    num_classes: int = 7
    batch_size: int = 32
    phase1_epochs: int = 20
    phase2_epochs: int = 40          # more epochs to recover from lower LR
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-5          # 10× lower than before — prevents destroying pretrained weights
    dropout: float = 0.5             # 0.4 → 0.5 to reduce phase-2 overfitting
    dense_units: int = 256
    unfreeze_layers: int = 30
    early_stop_patience: int = 12    # 8 → 12 to ride through LR warm-up
    l2_reg: float = 1e-4             # L2 weight decay on Dense layers
    label_smoothing: float = 0.1     # combats FER2013's noisy labels
    mixup_alpha: float = 0.2         # MixUp batch augmentation; 0.0 to disable
    save_path: Path = ROOT / "models" / "saved" / "mobilenet_finetuned.keras"
    data_dir: Path = ROOT / "archive"


CFG = TransferConfig()


def build_transfer_model(cfg: TransferConfig) -> keras.Model:
    """Build MobileNetV2-based model with frozen base.

    Args:
        cfg: TransferConfig hyper-parameter dataclass.

    Returns:
        Compiled Keras Model (base frozen).
    """
    base = MobileNetV2(
        input_shape=(cfg.img_size, cfg.img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inp = keras.Input(shape=(cfg.img_size, cfg.img_size, 3), name="face_input")
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        cfg.dense_units,
        kernel_regularizer=keras.regularizers.l2(cfg.l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(cfg.num_classes, activation="softmax", name="emotion_output")(x)

    model = keras.Model(inp, out, name="EmotionMobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.phase1_lr),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.label_smoothing
        ),
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: keras.Model, cfg: TransferConfig) -> None:
    """Unfreeze the last N layers of the MobileNetV2 base for fine-tuning.

    Args:
        model: The full transfer model.
        cfg: TransferConfig with unfreeze_layers and phase2_lr.
    """
    base = model.layers[1]  # MobileNetV2 base
    base.trainable = True
    for layer in base.layers[: -cfg.unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.phase2_lr),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.label_smoothing
        ),
        metrics=["accuracy"],
    )
    trainable_count = sum(1 for l in base.layers if l.trainable)
    logger.info("Unfroze last %d layers of MobileNetV2 (%d trainable layers total)",
                cfg.unfreeze_layers, trainable_count)


def train(cfg: TransferConfig = CFG) -> keras.callbacks.History:
    """Full two-phase training: load → phase1 → unfreeze → phase2 → save.

    Args:
        cfg: TransferConfig hyper-parameter dataclass.

    Returns:
        Keras History from phase 2 fine-tuning.
    """
    # ── Data loading ─────────────────────────────────────────────────────────
    logger.info("Loading and preprocessing dataset …")
    X_train_48, y_train, X_val_48, y_val, X_test_48, y_test = load_fer2013_from_folders(str(cfg.data_dir))

    logger.info("Upscaling to %dx%d RGB …", cfg.img_size, cfg.img_size)
    # preprocess_for_mobilenet returns [0, 1]; we keep val/test at the same
    # scale here and apply_mobilenet_scale to shift to [-1, 1] before eval.
    X_train_rgb = preprocess_for_mobilenet(X_train_48, cfg.img_size)
    X_val_rgb   = preprocess_for_mobilenet(X_val_48,   cfg.img_size)
    X_test_rgb  = preprocess_for_mobilenet(X_test_48,  cfg.img_size)

    # MobileNetV2 MUST receive inputs in [-1, 1].  Using [0, 1] causes the
    # model to see near-constant -1 values after the internal scale step and
    # results in ~10 % accuracy regardless of training length.
    X_val  = apply_mobilenet_scale(X_val_rgb)
    X_test = apply_mobilenet_scale(X_test_rgb)

    class_weights = compute_class_weights(y_train)

    # Use tf.data.Dataset — fixes the Keras 3.x ImageDataGenerator epoch-
    # boundary bug.  mobilenet_preprocess=True augments first (in [0,1]) then
    # shifts to [-1, 1] so augmentation arithmetic stays correct.
    train_ds = build_tf_dataset(
        X_train_rgb, y_train,
        batch_size=cfg.batch_size,
        augment=True,
        mobilenet_preprocess=True,
        mixup_alpha=cfg.mixup_alpha,
    )
    steps = max(1, len(X_train_rgb) // cfg.batch_size)

    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        str(cfg.save_path), monitor="val_accuracy",
        save_best_only=True, verbose=1,
    )
    es_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=cfg.early_stop_patience,
        restore_best_weights=True, verbose=1,
    )

    # ── Phase 1: train head only ─────────────────────────────────────────────
    model = build_transfer_model(cfg)
    model.summary(print_fn=logger.info)

    logger.info("=== Phase 1: training top layers (%d epochs) ===", cfg.phase1_epochs)
    model.fit(
        train_ds,
        steps_per_epoch=steps,
        epochs=cfg.phase1_epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[ckpt_cb, es_cb],
        verbose=1,
    )

    # ── Phase 2: fine-tune top N layers ─────────────────────────────────────
    unfreeze_top_layers(model, cfg)

    logger.info("=== Phase 2: fine-tuning last %d layers (%d epochs) ===",
                cfg.unfreeze_layers, cfg.phase2_epochs)
    # Build a fresh dataset iterator for phase 2 to reset the repeat cycle
    train_ds2 = build_tf_dataset(
        X_train_rgb, y_train,
        batch_size=cfg.batch_size,
        augment=True,
        mobilenet_preprocess=True,
        mixup_alpha=cfg.mixup_alpha,
    )
    history = model.fit(
        train_ds2,
        steps_per_epoch=steps,
        epochs=cfg.phase2_epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[ckpt_cb, es_cb],
        verbose=1,
    )

    logger.info("Fine-tuning complete. Model saved to %s", cfg.save_path)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f  Test loss: %.4f", acc, loss)

    return history


if __name__ == "__main__":
    train()

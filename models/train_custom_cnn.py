"""Train a custom CNN on FER2013 for emotion recognition."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# project root on sys.path so sibling packages resolve
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.dataset_loader import (
    build_tf_dataset,
    compute_class_weights,
    load_fer2013_from_folders,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
@dataclass
class CNNConfig:
    img_size: int = 48
    num_classes: int = 7
    batch_size: int = 64
    epochs: int = 100
    initial_lr: float = 1e-3
    dropout_conv: float = 0.25
    dropout_dense: float = 0.5
    dense_units: int = 512
    early_stop_patience: int = 15
    min_lr: float = 1e-6
    label_smoothing: float = 0.1   # combats FER2013's noisy labels
    mixup_alpha: float = 0.2       # MixUp batch augmentation; 0.0 to disable
    save_path: Path = ROOT / "models" / "saved" / "custom_cnn.keras"
    data_dir: Path = ROOT / "archive"


CFG = CNNConfig()


def build_custom_cnn(cfg: CNNConfig) -> keras.Model:
    """Construct the custom CNN with SE attention and residual connections.

    Architecture:
        Block 1 — 64 filters, plain double-conv → MaxPool
        Block 2 — 128 filters, residual + SE attention → MaxPool
        Block 3 — 256 filters, residual + SE attention → MaxPool (last_conv here)
        Head    — GlobalAveragePool → Dense(512) → Dense(7)

    SE (Squeeze-and-Excitation) blocks learn per-channel attention weights,
    boosting informative feature maps and suppressing noise — important for
    FER2013's low-resolution 48×48 images.

    Residual shortcuts help gradient flow and allow the network to skip
    unhelpful transformations, making deeper training more stable.

    Args:
        cfg: CNNConfig hyper-parameter dataclass.

    Returns:
        Compiled Keras Model.
    """
    inp = keras.Input(shape=(cfg.img_size, cfg.img_size, 1), name="face_input")

    # ── Block 1: 64 filters, plain double-conv ──────────────────────────────
    # 48×48 → 24×24 after MaxPool
    x = layers.Conv2D(64, 3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(cfg.dropout_conv)(x)

    # ── Block 2: 128 filters + residual + SE ────────────────────────────────
    # 24×24 → 12×12 after MaxPool
    shortcut = layers.Conv2D(128, 1, padding="same")(x)   # projection shortcut
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # SE attention: squeeze → excite → scale
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(16, activation="relu")(se)
    se = layers.Dense(128, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, 128))(se)
    x = layers.Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(cfg.dropout_conv)(x)

    # ── Block 3: 256 filters + residual + SE ────────────────────────────────
    # 12×12 → 6×6 after MaxPool  ("last_conv" is the Grad-CAM target)
    shortcut = layers.Conv2D(256, 1, padding="same")(x)   # projection shortcut
    shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, padding="same", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    # SE attention
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(32, activation="relu")(se)
    se = layers.Dense(256, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, 256))(se)
    x = layers.Multiply()([x, se])
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(cfg.dropout_conv)(x)

    # ── Dense head ──────────────────────────────────────────────────────────
    # GlobalAveragePooling reduces 6×6×256 → 256, fewer params than Flatten
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(cfg.dense_units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(cfg.dropout_dense)(x)
    out = layers.Dense(cfg.num_classes, activation="softmax", name="emotion_output")(x)

    model = keras.Model(inp, out, name="EmotionCNN")

    # Cosine decay LR schedule (non-settable → ReduceLROnPlateau must NOT be used)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg.initial_lr,
        decay_steps=cfg.epochs * 1000,
        alpha=cfg.min_lr / cfg.initial_lr,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        # Label smoothing reduces overconfidence on FER2013's noisy annotations
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=cfg.label_smoothing
        ),
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)
    return model


def train(cfg: CNNConfig = CFG) -> keras.callbacks.History:
    """Full training run: load data → build model → fit → save.

    Args:
        cfg: CNNConfig hyper-parameter dataclass.

    Returns:
        Keras History object.
    """
    # Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013_from_folders(str(cfg.data_dir))
    class_weights = compute_class_weights(y_train)

    # Use tf.data.Dataset — fixes the Keras 3.x ImageDataGenerator epoch-boundary
    # bug where every other epoch only receives one batch of data.
    train_ds = build_tf_dataset(
        X_train, y_train,
        batch_size=cfg.batch_size,
        augment=True,
        mixup_alpha=cfg.mixup_alpha,
    )

    # Model
    model = build_custom_cnn(cfg)
    # NOTE: ReduceLROnPlateau is intentionally omitted.  The CosineDecay
    # schedule baked into the Adam optimizer is non-settable in Keras 3.x,
    # so ReduceLROnPlateau would crash at epoch 6 with a TypeError.
    # CosineDecay already handles LR annealing automatically.

    # Callbacks
    cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(cfg.save_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    steps_per_epoch = max(1, len(X_train) // cfg.batch_size)
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training complete. Model saved to %s", cfg.save_path)

    # Quick test-set evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f  Test loss: %.4f", acc, loss)

    return history


if __name__ == "__main__":
    train()

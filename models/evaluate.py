"""Evaluate both trained models on the FER2013 test set and produce reports."""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from utils.dataset_loader import load_fer2013_from_folders, preprocess_for_mobilenet
from utils.metrics_plot import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_training_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
SAVED_DIR = ROOT / "models" / "saved"
ARCHIVE_DIR = ROOT / "archive"


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_oh: np.ndarray,
    model_name: str,
) -> Dict:
    """Run inference on test set and compute all metrics.

    Args:
        model: Loaded Keras model.
        X_test: Test images already preprocessed for this model.
        y_test_oh: One-hot test labels, shape (N, 7).
        model_name: Display name for logging and file naming.

    Returns:
        Dict with accuracy, macro_f1, and per-class report dict.
    """
    logger.info("Evaluating %s on %d test samples …", model_name, len(X_test))
    y_pred_prob = model.predict(X_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_oh, axis=1)

    _, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    report = classification_report(
        y_true, y_pred,
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )
    macro_f1 = report["macro avg"]["f1-score"]

    safe_name = model_name.lower().replace(" ", "_")
    plot_confusion_matrix(
        y_true, y_pred,
        model_name=model_name,
        save_path=str(SAVED_DIR / f"cm_{safe_name}.png"),
    )
    plot_per_class_metrics(
        report, model_name=model_name,
        save_path=str(SAVED_DIR / f"metrics_{safe_name}.png"),
    )

    logger.info("%s → Accuracy: %.4f  Macro-F1: %.4f", model_name, acc, macro_f1)
    print(f"\n{'=' * 55}")
    print(f"  {model_name}  |  Accuracy: {acc:.4f}  |  Macro-F1: {macro_f1:.4f}")
    print(f"{'=' * 55}")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS, zero_division=0))

    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "report": report}


def run_evaluation() -> None:
    """Load both models and run full evaluation pipeline."""
    SAVED_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    _, _, _, _, X_test_gray, y_test = load_fer2013_from_folders(str(ARCHIVE_DIR))

    results: Dict[str, Dict] = {}

    # ── Custom CNN ───────────────────────────────────────────────────────────
    cnn_path = SAVED_DIR / "custom_cnn.keras"
    if cnn_path.exists():
        cnn = keras.models.load_model(str(cnn_path))
        results["Custom CNN"] = evaluate_model(cnn, X_test_gray, y_test, "Custom CNN")
    else:
        logger.warning("Custom CNN not found at %s — skipping. Run train_custom_cnn.py first.", cnn_path)

    # ── MobileNetV2 ──────────────────────────────────────────────────────────
    mobile_path = SAVED_DIR / "mobilenet_finetuned.keras"
    if mobile_path.exists():
        mobile = keras.models.load_model(str(mobile_path))
        X_test_rgb = preprocess_for_mobilenet(X_test_gray, target_size=96)
        results["MobileNetV2"] = evaluate_model(mobile, X_test_rgb, y_test, "MobileNetV2")
    else:
        logger.warning("MobileNetV2 not found at %s — skipping. Run train_transfer.py first.", mobile_path)

    if not results:
        logger.error("No trained models found. Train at least one model before evaluating.")
        return

    # Save JSON
    output_json = SAVED_DIR / "evaluation_results.json"
    serialisable = {
        name: {k: v for k, v in data.items() if k != "report"}
        for name, data in results.items()
    }
    with open(output_json, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Evaluation results saved to %s", output_json)


if __name__ == "__main__":
    run_evaluation()

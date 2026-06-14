"""Grad-CAM heatmap generation for emotion CNN models."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

HEATMAP_ALPHA = 0.40   # opacity of the color overlay blended on the face crop
COLORMAP = cv2.COLORMAP_JET


def get_last_conv_layer_name(model: keras.Model) -> Optional[str]:
    """Auto-detect the name of the last *feature-extracting* Conv2D layer.

    1×1 convolutions are intentionally skipped because they are typically
    projection shortcuts in residual blocks (dimension-matching only) and
    produce poor Grad-CAM heatmaps.  The last 3×3 (or larger) Conv2D is
    preferred as the Grad-CAM target.

    Args:
        model: A Keras Model instance.

    Returns:
        Layer name string, or None if no suitable Conv2D layer is found.
    """
    def _is_feature_conv(layer: keras.layers.Layer) -> bool:
        """Return True for Conv2D layers with kernel > 1×1."""
        if not isinstance(layer, keras.layers.Conv2D):
            return False
        ks = layer.kernel_size
        return ks != (1, 1) and ks != [1, 1]

    last_name = None
    for layer in model.layers:
        if _is_feature_conv(layer):
            last_name = layer.name
        # Handle nested models (e.g. MobileNetV2 sub-model)
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if _is_feature_conv(sub):
                    last_name = sub.name
    return last_name


def _get_grad_model(
    model: keras.Model,
    conv_layer_name: Optional[str] = None,
) -> Tuple[keras.Model, str]:
    """Build a Grad-CAM gradient model targeting the specified conv layer.

    Always uses the FULL model's output (shape (batch, 7)) so that
    predictions[:, class_idx] is valid regardless of model type (CNN or
    MobileNetV2 which has a nested sub-model).

    Args:
        model: Original Keras Model.
        conv_layer_name: Name of the target conv layer; auto-detected if None.

    Returns:
        Tuple of (gradient_model, resolved_layer_name).

    Raises:
        ValueError: If the layer name cannot be resolved.
    """
    if conv_layer_name is None:
        conv_layer_name = get_last_conv_layer_name(model)
    if conv_layer_name is None:
        raise ValueError("Could not find any Conv2D layer in the model.")

    # Find the conv output tensor that is reachable from the outer model's inputs.
    #
    # For a flat CNN this is straightforward: model.get_layer(name).output.
    #
    # For MobileNetV2 (or any nested functional sub-model) the named Conv2D
    # lives inside the sub-model's INTERNAL graph, whose tensors are connected
    # to *sub-model.inputs* — NOT to the outer model's inputs.  Passing such a
    # tensor as an output of keras.models.Model(..., inputs=model.inputs, ...)
    # would create a disconnected graph, causing the two grad-model outputs to
    # be mis-ordered at runtime (predictions ends up as the 4-D conv feature
    # map rather than the (batch, 7) logits, so predictions[:, class_idx]
    # raises "slice index N of dimension 1 out of bounds").
    #
    # The fix: when the layer is nested, use the SUB-MODEL LAYER's own output
    # tensor (which IS wired into the outer model's graph), not an internal
    # tensor from inside the sub-model.  For MobileNetV2 this gives the last
    # spatial feature map (None, 3, 3, 1280), which is a perfectly valid
    # Grad-CAM target.
    conv_output = None
    try:
        conv_output = model.get_layer(conv_layer_name).output
    except ValueError:
        for layer in model.layers:
            if not hasattr(layer, "layers"):
                continue
            try:
                layer.get_layer(conv_layer_name)        # verify the layer exists here
                conv_output = layer.output              # ← outer-model–connected tensor
                break
            except ValueError:
                pass

    if conv_output is None:
        raise ValueError(f"Layer '{conv_layer_name}' not found in model or sub-models.")

    # Both tensors are now reachable from model.inputs, so Keras can trace the
    # full graph and predictions will always have shape (batch, num_classes).
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[conv_output, model.output],
    )
    return grad_model, conv_layer_name


def generate_heatmap(
    model: keras.Model,
    face_crop: np.ndarray,
    predicted_class: int,
    conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate a Grad-CAM RGBA overlay for a single face crop.

    Args:
        model: Trained Keras Model (Custom CNN or MobileNetV2).
        face_crop: Preprocessed face input matching the model's expected shape
                   (H, W, C) float32 — do NOT include the batch dim.
                   CNN expects ``[0, 1]``; MobileNetV2 expects ``[-1, 1]``.
        predicted_class: Class index (0–6) to explain.
        conv_layer_name: Name of the target Conv2D; auto-detected if None.

    Returns:
        RGBA uint8 array of shape (H, W, 4) — Grad-CAM overlay ready to blend.
    """
    h, w = face_crop.shape[:2]

    try:
        grad_model, resolved_name = _get_grad_model(model, conv_layer_name)
    except ValueError as exc:
        logger.warning("Grad-CAM setup failed: %s. Returning empty overlay.", exc)
        return np.zeros((h, w, 4), dtype=np.uint8)

    inp_batch = np.expand_dims(face_crop, axis=0)
    inp_tensor = tf.cast(inp_batch, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inp_tensor)
        conv_outputs, predictions = grad_model(inp_tensor)
        class_channel = predictions[:, predicted_class]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        logger.warning("Grad-CAM: gradient is None for layer '%s'.", resolved_name)
        return np.zeros((h, w, 4), dtype=np.uint8)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Normalise
    heatmap = np.maximum(heatmap, 0)
    denom = heatmap.max()
    if denom > 0:
        heatmap /= denom

    # Resize to original face size
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    colored = cv2.applyColorMap(heatmap_resized, COLORMAP)  # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    # Build RGBA with alpha = heatmap intensity
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = colored_rgb
    rgba[:, :, 3] = (heatmap_resized * HEATMAP_ALPHA * 255 / heatmap_resized.max()
                     if heatmap_resized.max() > 0 else 0)
    return rgba


def blend_heatmap_on_face(
    face_bgr: np.ndarray,
    heatmap_rgba: np.ndarray,
) -> np.ndarray:
    """Alpha-blend a Grad-CAM RGBA overlay onto a BGR face image.

    Args:
        face_bgr: BGR uint8 face crop, shape (H, W, 3).
        heatmap_rgba: RGBA uint8 Grad-CAM overlay, shape (H, W, 4).

    Returns:
        Blended BGR uint8 image, shape (H, W, 3).
    """
    alpha = heatmap_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = cv2.cvtColor(heatmap_rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    base = face_bgr.astype(np.float32)
    blended = base * (1 - alpha) + overlay_bgr * alpha
    return blended.clip(0, 255).astype(np.uint8)

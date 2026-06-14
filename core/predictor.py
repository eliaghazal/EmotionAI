"""Real-time emotion inference pipeline with temporal smoothing, Grad-CAM,
face recognition, and on-frame annotation."""

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tensorflow import keras

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTION_LABELS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]
EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Angry":    (68, 68, 255),
    "Disgust":  (182, 89, 155),
    "Fear":     (34, 126, 230),
    "Happy":    (15, 196, 241),
    "Sad":      (219, 152, 52),
    "Surprise": (156, 188, 26),
    "Neutral":  (166, 165, 149),
}
KNOWN_COLOR  = (50, 205, 50)    # green for known faces
UNKNOWN_COLOR = (0, 165, 255)   # orange for unknown faces

SMOOTH_WINDOW = 3               # frames for temporal smoothing (lower = more responsive)
TEMPERATURE   = 4.0             # softmax temperature: higher = softer distribution across emotions
TARGET_FPS    = 15
CNN_INPUT_SIZE   = 48
MOBILE_INPUT_SIZE = 96
RECOMPUTE_IOU_THRESHOLD = 0.70  # re-run face_recognition if face moved a lot
BAR_HEIGHT = 8
BAR_SPACING = 12
LABEL_FONT_SCALE = 0.55
LABEL_THICKNESS = 1
BOX_THICKNESS = 2
CORNER_RADIUS = 8


def _iou(a: Tuple, b: Tuple) -> float:
    """Compute IoU between two (x,y,w,h) boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _draw_rounded_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    radius: int = CORNER_RADIUS,
) -> None:
    """Draw a rounded-corner rectangle on img in place."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def _draw_pill_label(
    img: np.ndarray,
    text: str,
    cx: int, cy: int,
    bg_color: Tuple[int, int, int],
) -> None:
    """Draw a pill-shaped label badge centered at (cx, cy)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)
    pad_x, pad_y = 8, 4
    x1, y1 = cx - tw // 2 - pad_x, cy - th // 2 - pad_y
    x2, y2 = cx + tw // 2 + pad_x, cy + th // 2 + pad_y
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    _draw_rounded_rect(img, x1, y1, x2, y2, (255, 255, 255), 1, radius=min(6, (y2 - y1) // 2))
    cv2.putText(img, text, (x1 + pad_x, y2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (255, 255, 255), LABEL_THICKNESS)


@dataclass
class _FaceState:
    """Temporal smoothing buffer and embedding cache for a tracked face."""
    smooth_buffer: deque = field(default_factory=lambda: deque(maxlen=SMOOTH_WINDOW))
    last_bbox: Optional[Tuple] = None
    last_embedding: Optional[np.ndarray] = None
    person_name: Optional[str] = None
    recognition_confidence: float = 0.0


class EmotionPredictor:
    """End-to-end emotion recognition and annotation pipeline.

    Args:
        model: Loaded Keras model (Custom CNN or MobileNetV2).
        model_type: "cnn" for Custom CNN, "mobilenet" for MobileNetV2.
        registry: Optional FaceRegistry for identity lookup.
        ensemble_model: Optional second Keras model to average with the primary.
            When provided, probabilities from both models are averaged before
            temperature scaling, improving robustness.
        ensemble_model_type: "cnn" or "mobilenet" — type of the ensemble model.
    """

    def __init__(
        self,
        model: keras.Model,
        model_type: str = "cnn",
        registry=None,
        ensemble_model: Optional[keras.Model] = None,
        ensemble_model_type: Optional[str] = None,
    ) -> None:
        self._model = model
        self._model_type = model_type.lower()
        self._input_size = CNN_INPUT_SIZE if self._model_type == "cnn" else MOBILE_INPUT_SIZE
        self._grayscale = self._model_type == "cnn"
        self._registry = registry
        self._ensemble_model = ensemble_model
        self._ensemble_model_type = ensemble_model_type
        self._face_states: Dict[int, _FaceState] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._last_fps_time = time.time()
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._face_recognition = None
        self._face_recognition_disabled = False
        # CLAHE for contrast normalisation on webcam input (avoids distribution shift vs FER2013)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Import Grad-CAM lazily
        from core.gradcam import generate_heatmap, blend_heatmap_on_face, get_last_conv_layer_name
        self._gen_heatmap = generate_heatmap
        self._blend_heatmap = blend_heatmap_on_face
        self._last_conv = get_last_conv_layer_name(model)

        if ensemble_model is not None:
            logger.info(
                "EmotionPredictor: ensemble enabled (%s + %s)",
                model_type, ensemble_model_type,
            )

    def switch_model(
        self,
        model: keras.Model,
        model_type: str,
        ensemble_model: Optional[keras.Model] = None,
        ensemble_model_type: Optional[str] = None,
    ) -> None:
        """Hot-swap the underlying Keras model (and optional ensemble) at runtime.

        Args:
            model: New primary Keras model.
            model_type: "cnn" or "mobilenet".
            ensemble_model: Optional second model to average with the primary.
            ensemble_model_type: "cnn" or "mobilenet" — type of ensemble model.
        """
        from core.gradcam import get_last_conv_layer_name
        self._model = model
        self._model_type = model_type.lower()
        self._input_size = CNN_INPUT_SIZE if self._model_type == "cnn" else MOBILE_INPUT_SIZE
        self._grayscale = self._model_type == "cnn"
        self._last_conv = get_last_conv_layer_name(model)
        self._ensemble_model = ensemble_model
        self._ensemble_model_type = ensemble_model_type
        self._face_states.clear()
        # MobileNet uses RGB so CLAHE is applied per-channel; re-create to be safe
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ens_info = f" + ensemble={ensemble_model_type}" if ensemble_model else ""
        logger.info("EmotionPredictor switched to model_type='%s'%s", model_type, ens_info)

    def _preprocess_for_model(
        self, crop: np.ndarray, model_type: str
    ) -> np.ndarray:
        """Preprocess a BGR crop for a specific model type.

        * ``"cnn"``      → grayscale + CLAHE, ``[0, 1]``, shape ``(48, 48, 1)``
        * ``"mobilenet"``→ RGB, ``[-1, 1]``, shape ``(96, 96, 3)``

        Args:
            crop: BGR uint8 face crop.
            model_type: ``"cnn"`` or ``"mobilenet"``.

        Returns:
            Float32 array in the range expected by that model.
        """
        if model_type.lower() == "cnn":
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = self._clahe.apply(gray)
            resized = cv2.resize(gray, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            return (resized.astype(np.float32) / 255.0)[..., np.newaxis]
        else:
            resized = cv2.resize(crop, (MOBILE_INPUT_SIZE, MOBILE_INPUT_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            return (rgb.astype(np.float32) / 255.0) * 2.0 - 1.0

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalise a BGR crop for the **active** model.

        Delegates to :meth:`_preprocess_for_model` using the currently active
        model type.  Called directly by ``api_gradcam`` in ``web/app.py``.

        Args:
            crop: BGR uint8 face crop.

        Returns:
            Float32 array of shape ``(H, W, C)`` in the range expected by the
            active model.
        """
        return self._preprocess_for_model(crop, self._model_type)

    def _infer(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference (with optional ensemble averaging) on a single face.

        Preprocesses the crop for the primary model (and ensemble model if
        configured), averages the raw softmax probabilities, then applies
        temperature scaling to soften the distribution.

        Args:
            crop_bgr: BGR uint8 face crop.

        Returns:
            Tuple of:
            * ``scores``     — calibrated softmax probabilities, shape (7,).
            * ``face_input`` — preprocessed crop for the primary model, used
                               by Grad-CAM in :meth:`process_frame`.
        """
        face_input = self._preprocess_crop(crop_bgr)
        batch = np.expand_dims(face_input, 0)
        probs = self._model.predict(batch, verbose=0)[0]

        # Ensemble: average with second model's probabilities if available
        if self._ensemble_model is not None and self._ensemble_model_type is not None:
            ens_input = self._preprocess_for_model(crop_bgr, self._ensemble_model_type)
            ens_batch = np.expand_dims(ens_input, 0)
            ens_probs = self._ensemble_model.predict(ens_batch, verbose=0)[0]
            probs = (probs + ens_probs) / 2.0

        # Temperature scaling: recover logits, divide by T, re-softmax
        logits = np.log(np.clip(probs, 1e-7, 1.0))
        scaled = logits / TEMPERATURE
        scaled -= scaled.max()          # numerical stability
        exp = np.exp(scaled)
        return (exp / exp.sum()).astype(np.float32), face_input

    def _smooth(self, face_id: int, scores: np.ndarray) -> np.ndarray:
        """Apply temporal weighted average smoothing.

        Args:
            face_id: Unique face track identifier.
            scores: Raw softmax scores, shape (7,).

        Returns:
            Smoothed probability array, shape (7,).
        """
        state = self._face_states.setdefault(face_id, _FaceState())
        state.smooth_buffer.append(scores)
        weights = np.linspace(0.5, 1.0, len(state.smooth_buffer))
        weighted = np.average(np.stack(list(state.smooth_buffer)), axis=0, weights=weights)
        return weighted

    def _get_or_compute_embedding(
        self, face_id: int, crop: np.ndarray, bbox: Tuple
    ) -> Optional[np.ndarray]:
        """Return cached embedding or compute a new one if face moved.

        Args:
            face_id: Face track ID.
            crop: BGR face crop.
            bbox: Current (x, y, w, h) bounding box.

        Returns:
            128-d face embedding or None if face_recognition unavailable.
        """
        if self._face_recognition_disabled:
            return None

        state = self._face_states.setdefault(face_id, _FaceState())
        if (state.last_bbox is not None
                and _iou(state.last_bbox, bbox) >= RECOMPUTE_IOU_THRESHOLD
                and state.last_embedding is not None):
            return state.last_embedding

        try:
            if self._face_recognition is None:
                import face_recognition
                self._face_recognition = face_recognition
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = self._face_recognition.face_encodings(rgb)
            emb = np.array(encs[0]) if encs else None
        except SystemExit as exc:
            self._face_recognition_disabled = True
            logger.warning("Face recognition disabled: %s", exc)
            emb = None
        except Exception as exc:
            self._face_recognition_disabled = True
            logger.warning("Face recognition disabled: %s", exc)
            emb = None

        state.last_bbox = bbox
        state.last_embedding = emb
        return emb

    def process_frame(
        self,
        frame: np.ndarray,
        detections: list,
        show_gradcam: bool = False,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Process one video frame: infer, annotate, and return results.

        Args:
            frame: BGR uint8 frame from the camera.
            detections: List of FaceDetection objects from the detector.
            show_gradcam: Blend Grad-CAM heatmap onto each face region.

        Returns:
            Tuple of (annotated_frame, results_list).
            Each element of results_list is a dict with:
                face_id, emotion_label, confidence_scores, face_bbox,
                person_name, recognition_confidence.
        """
        annotated = frame.copy()
        results: List[Dict] = []
        known_count = 0
        unknown_count = 0

        for idx, det in enumerate(detections):
            face_id = idx  # simple ID; production uses Kalman/IoU tracker
            bbox = (det.x, det.y, det.w, det.h)

            # Crop
            y1 = max(0, det.y)
            y2 = min(frame.shape[0], det.y + det.h)
            x1 = max(0, det.x)
            x2 = min(frame.shape[1], det.x + det.w)
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            # Preprocess & infer (face_input kept for Grad-CAM below)
            raw_scores, face_input = self._infer(crop_bgr)
            scores = self._smooth(face_id, raw_scores)
            pred_idx = int(np.argmax(scores))
            emotion = EMOTION_LABELS[pred_idx]
            confidence = float(scores[pred_idx])

            # Face recognition
            state = self._face_states.setdefault(face_id, _FaceState())
            if self._registry is not None:
                emb = self._get_or_compute_embedding(face_id, crop_bgr, bbox)
                if emb is not None:
                    name, rec_conf = self._registry.identify_face(emb)
                    state.person_name = name
                    state.recognition_confidence = rec_conf
                elif state.last_embedding is None:
                    # face_recognition couldn't encode this face (e.g. too small,
                    # partial, or just appeared) — don't carry over a stale identity
                    # from whatever occupied this face_id slot before.
                    state.person_name = None
                    state.recognition_confidence = 0.0
            person_name = state.person_name
            rec_conf = state.recognition_confidence

            if person_name:
                known_count += 1
            else:
                unknown_count += 1

            # Grad-CAM overlay
            if show_gradcam:
                try:
                    hm = self._gen_heatmap(self._model, face_input, pred_idx, self._last_conv)
                    hm_resized = cv2.resize(hm, (x2 - x1, y2 - y1))
                    blended = self._blend_heatmap(crop_bgr, hm_resized)
                    annotated[y1:y2, x1:x2] = blended
                except Exception as exc:
                    logger.debug("Grad-CAM failed: %s", exc)

            # ── Draw annotations ───────────────────────────────────────────
            box_color = KNOWN_COLOR if person_name else UNKNOWN_COLOR
            emo_color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            _draw_rounded_rect(annotated, x1, y1, x2, y2, box_color, BOX_THICKNESS)

            # Name label (above box)
            label_y = max(y1 - 24, 14)
            if person_name:
                name_text = f"{person_name} ({rec_conf:.0%})"
                _draw_pill_label(annotated, name_text, (x1 + x2) // 2, label_y, KNOWN_COLOR)
                emo_y = label_y + 20
            else:
                _draw_pill_label(annotated, "? Unknown", (x1 + x2) // 2, label_y, UNKNOWN_COLOR)
                emo_y = label_y + 20

            # Emotion badge (just above or inside box top)
            emo_text = f"{emotion} {confidence:.0%}"
            _draw_pill_label(annotated, emo_text, (x1 + x2) // 2, emo_y, emo_color)

            # Mini confidence bars for all 7 classes
            bar_x = x1 + 4
            bar_y = y2 - (len(EMOTION_LABELS) * BAR_SPACING) - 4
            for i, (emo, s) in enumerate(zip(EMOTION_LABELS, scores)):
                bx1 = bar_x
                by1 = bar_y + i * BAR_SPACING
                bar_w = int((x2 - x1 - 8) * s)
                col = EMOTION_COLORS.get(emo, (200, 200, 200))
                cv2.rectangle(annotated, (bx1, by1), (bx1 + bar_w, by1 + BAR_HEIGHT), col, -1)
                cv2.putText(annotated, emo[:3], (bx1, by1 + BAR_HEIGHT - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

            results.append({
                "face_id": face_id,
                "emotion_label": emotion,
                "confidence_scores": scores.tolist(),
                "face_bbox": bbox,
                "person_name": person_name,
                "recognition_confidence": float(rec_conf),
                # embedding is needed by the registration queue in web/app.py
                "embedding": state.last_embedding,
            })

        # ── HUD overlay ────────────────────────────────────────────────────────
        hud_text = f"Known: {known_count}  Unknown: {unknown_count}"
        cv2.putText(annotated, hud_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS counter
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now
        cv2.putText(annotated, f"FPS:{self._fps:.1f}", (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return annotated, results

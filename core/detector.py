"""Face detection: MediaPipe Tasks API primary, OpenCV Haar Cascade fallback."""

import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
# Haar parameters tuned to minimise false positives on webcam input
HAAR_SCALE_FACTOR = 1.15
HAAR_MIN_NEIGHBORS = 8
HAAR_MIN_SIZE = (60, 60)
PAD_RATIO = 0.10            # padding around each face crop as fraction of box size

MEDIAPIPE_MODEL_PATH = Path(__file__).parent.parent / "models" / "blaze_face_short_range.tflite"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _ensure_mediapipe_model() -> Optional[Path]:
    """Download the BlazeFace model file if not present. Returns path or None."""
    if MEDIAPIPE_MODEL_PATH.exists():
        return MEDIAPIPE_MODEL_PATH
    try:
        MEDIAPIPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading BlazeFace model …")
        urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, MEDIAPIPE_MODEL_PATH)
        logger.info("BlazeFace model saved to %s", MEDIAPIPE_MODEL_PATH)
        return MEDIAPIPE_MODEL_PATH
    except Exception as exc:
        logger.warning("Could not download BlazeFace model: %s", exc)
        return None


@dataclass
class FaceDetection:
    """A single detected face with all associated data."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    track_id: Optional[int] = None


class FaceDetector:
    """Detect faces using MediaPipe Tasks API with OpenCV Haar Cascade fallback.

    Args:
        min_confidence: Minimum detection confidence score.
        fallback_to_haar: Use OpenCV Haar if MediaPipe is unavailable.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE,
        fallback_to_haar: bool = True,
        # kept for backward compat, no longer used
        use_full_range: bool = False,
    ) -> None:
        self._min_confidence = min_confidence
        self._mp_detector = None
        self._haar_cascade: Optional[cv2.CascadeClassifier] = None
        self._active_backend = "none"

        self._init_mediapipe()
        if self._mp_detector is None and fallback_to_haar:
            self._load_haar()

    # ── Initialisation ─────────────────────────────────────────────────────────
    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_mediapipe_model()
            if model_path is None:
                raise RuntimeError("BlazeFace model unavailable")

            options = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=self._min_confidence,
            )
            self._mp_detector = mp_vision.FaceDetector.create_from_options(options)
            self._active_backend = "mediapipe"
            self._mp = mp   # keep reference for mp.Image
            logger.info("FaceDetector: using MediaPipe Tasks BlazeFace")
        except Exception as exc:
            logger.warning("MediaPipe Tasks unavailable (%s). Trying Haar Cascade.", exc)

    def _load_haar(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        if self._haar_cascade.empty():
            logger.error("Haar Cascade failed to load from %s", cascade_path)
        else:
            self._active_backend = "haar"
            logger.info("FaceDetector: using Haar Cascade fallback")

    # ── Public API ─────────────────────────────────────────────────────────────
    @property
    def backend(self) -> str:
        return self._active_backend

    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect all faces in a BGR frame.

        Returns:
            List of FaceDetection objects sorted by descending confidence.
        """
        if self._active_backend == "mediapipe":
            return self._detect_mediapipe(frame)
        elif self._active_backend == "haar":
            return self._detect_haar(frame)
        logger.error("No face detection backend available.")
        return []

    # ── Backends ───────────────────────────────────────────────────────────────
    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceDetection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._mp_detector.detect(mp_image)
        faces: List[FaceDetection] = []
        h, w = frame.shape[:2]

        for detection in result.detections:
            bb = detection.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            bw = min(bb.width, w - x)
            bh = min(bb.height, h - y)
            score = detection.categories[0].score if detection.categories else 0.0
            if score >= self._min_confidence and bw > 0 and bh > 0:
                faces.append(FaceDetection(x=x, y=y, w=bw, h=bh, confidence=float(score)))

        return sorted(faces, key=lambda f: f.confidence, reverse=True)

    def _detect_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(gray, gray)   # normalise contrast before detection
        rects = self._haar_cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )
        if len(rects) == 0:
            return []
        rects = self._nms(rects)
        return [
            FaceDetection(x=int(x), y=int(y), w=int(w), h=int(h), confidence=0.9)
            for (x, y, w, h) in rects
        ]

    @staticmethod
    def _nms(rects, iou_threshold: float = 0.3) -> list:
        """Non-maximum suppression to remove overlapping Haar detections."""
        if len(rects) == 0:
            return []
        boxes = np.array([[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in rects],
                         dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas)
        keep = []
        while len(order):
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou < iou_threshold]
        return [rects[i] for i in keep]

    # ── Crop helper ────────────────────────────────────────────────────────────
    def crop_and_resize(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        target_size: int,
        grayscale: bool = True,
        pad: bool = True,
    ) -> np.ndarray:
        """Crop a face region and resize to (target_size, target_size)."""
        h, w = frame.shape[:2]
        px = int(face.w * PAD_RATIO) if pad else 0
        py = int(face.h * PAD_RATIO) if pad else 0
        x1 = max(0, face.x - px)
        y1 = max(0, face.y - py)
        x2 = min(w, face.x + face.w + px)
        y2 = min(h, face.y + face.h + py)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame[face.y:face.y + face.h, face.x:face.x + face.w]

        if grayscale:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(crop, (target_size, target_size))
            return (resized.astype(np.float32) / 255.0)[..., np.newaxis]
        else:
            resized = cv2.resize(crop, (target_size, target_size))
            return (resized.astype(np.float32) / 255.0)

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._mp_detector is not None:
            try:
                self._mp_detector.close()
            except Exception:
                pass
            self._mp_detector = None

    def __del__(self) -> None:
        self.close()

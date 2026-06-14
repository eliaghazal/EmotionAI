"""Flask + SocketIO backend for the real-time emotion recognition dashboard."""

import base64
import io
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from flask_socketio import SocketIO, emit

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.detector import FaceDetector
from core.face_registry import FaceRegistry
from core.predictor import EmotionPredictor
from core.session_tracker import SessionTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SAVED_DIR = ROOT / "models" / "saved"
SAMPLE_VIDEO = ROOT / "data" / "sample.mp4"
MJPEG_QUALITY = 80          # JPEG encode quality for video feed
STATS_EMIT_INTERVAL = 0.5   # seconds between SocketIO stat pushes
REGISTRATION_DELAY = 3.0    # seconds before prompting to register new faces
UNKNOWN_RE_QUEUE_INTERVAL = 300.0  # don't re-queue same unknown within 5 min

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "emotion_ai_secret"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global state (all access serialized by _state_lock) ────────────────────────
_state_lock = threading.Lock()
_camera: Optional[cv2.VideoCapture] = None
_detector: Optional[FaceDetector] = None
_predictor: Optional[EmotionPredictor] = None
_session_tracker: Optional[SessionTracker] = None
_registry: Optional[FaceRegistry] = None
_current_frame: Optional[np.ndarray] = None   # annotated frame (for MJPEG)
_raw_frame: Optional[np.ndarray] = None       # raw frame (for Grad-CAM)
_current_model_type = "cnn"
_show_gradcam = False
_stream_active = False
_browser_session_start = time.time()
_browser_frame_lock = threading.Lock()

# Model cache — keeps both models in memory after first load so switching
# between them (and ensemble inference) does not incur repeated disk reads.
_model_cache: dict = {}

# Registration state machine
_reg_queue: list = []          # list of {face_crop_b64, embedding, queued_at}
_reg_processing = False
_skipped_embeddings: dict = {} # embedding hash -> timestamp

# Emit stats thread
_stats_thread: Optional[threading.Thread] = None


def _process_dashboard_frame(frame: np.ndarray, session_start: float) -> np.ndarray:
    """Run detection, prediction, registration queueing, and stats for one frame."""
    global _current_frame, _raw_frame, _reg_queue, _reg_processing

    detections = _detector.detect_faces(frame) if _detector else []

    if _predictor:
        annotated, results = _predictor.process_frame(
            frame, detections, show_gradcam=_show_gradcam
        )
        _session_tracker.add_frame(results)

        if time.time() - session_start > REGISTRATION_DELAY and not _reg_processing:
            _maybe_queue_unknowns(frame, results)
    else:
        annotated = frame

    with _state_lock:
        _raw_frame = frame.copy()
        _current_frame = annotated.copy()

    return annotated


def _load_model(model_type: str):
    """Load a trained Keras model from disk.

    Args:
        model_type: "cnn" or "mobilenet".

    Returns:
        Keras Model or None if file not found.
    """
    from tensorflow import keras
    paths = {
        "cnn": SAVED_DIR / "custom_cnn.keras",
        "mobilenet": SAVED_DIR / "mobilenet_finetuned.keras",
    }
    path = paths.get(model_type)
    if path and path.exists():
        logger.info("Loading model from %s …", path)
        return keras.models.load_model(str(path))
    logger.warning("Model file not found: %s", path)
    return None


def _get_or_load_model(model_type: str):
    """Return a cached model, loading from disk on first access.

    Both models are kept in ``_model_cache`` so that switching the active
    model or enabling ensemble inference never blocks the video stream with
    repeated I/O.

    Args:
        model_type: "cnn" or "mobilenet".

    Returns:
        Keras Model or None if the file is not found.
    """
    if model_type not in _model_cache:
        model = _load_model(model_type)
        if model is not None:
            _model_cache[model_type] = model
            logger.info("Model '%s' cached.", model_type)
    return _model_cache.get(model_type)


def _open_camera() -> cv2.VideoCapture:
    """Open webcam or fall back to a sample video file."""
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        logger.info("Webcam opened (device 0).")
        return cap
    logger.warning("Webcam not found. Falling back to sample video.")
    if SAMPLE_VIDEO.exists():
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        if cap.isOpened():
            return cap
    raise RuntimeError(
        "No webcam and no sample video found at data/sample.mp4. "
        "Connect a webcam or place a sample.mp4 in the data/ directory."
    )


def _init_pipeline() -> None:
    """Initialize detector, predictor, tracker, and registry.

    Attempts to load BOTH models at startup so ensemble inference is available
    immediately.  If only one model is trained, single-model inference is used.
    """
    global _camera, _detector, _predictor, _session_tracker, _registry, _stream_active

    _registry = FaceRegistry()
    _detector = FaceDetector()
    _session_tracker = SessionTracker()

    model = _get_or_load_model(_current_model_type)
    other_type = "mobilenet" if _current_model_type == "cnn" else "cnn"
    ensemble_model = _get_or_load_model(other_type)

    if model is None:
        logger.warning(
            "No trained model found. Emotion labels will default to 'Neutral'. "
            "Train a model first: python models/train_custom_cnn.py"
        )
        _predictor = None
    elif ensemble_model is not None:
        logger.info(
            "Both models available — starting with ensemble inference "
            "(%s primary + %s ensemble).", _current_model_type, other_type
        )
        _predictor = EmotionPredictor(
            model, _current_model_type, _registry,
            ensemble_model=ensemble_model,
            ensemble_model_type=other_type,
        )
    else:
        _predictor = EmotionPredictor(model, _current_model_type, _registry)

    try:
        _camera = _open_camera()
        _stream_active = True
    except RuntimeError as exc:
        logger.error(str(exc))
        _stream_active = False


def _frame_generator():
    """Generator that yields MJPEG-encoded annotated frames."""
    session_start = time.time()

    while True:
        if not _stream_active or _camera is None:
            time.sleep(0.05)
            continue

        ret, frame = _camera.read()
        if not ret:
            if isinstance(_camera, cv2.VideoCapture):
                _camera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop sample video
            continue

        annotated = _process_dashboard_frame(frame, session_start)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


def _maybe_queue_unknowns(frame: np.ndarray, results: list) -> None:
    """Add unknown faces to the registration queue if not already queued."""
    global _reg_queue
    now = time.time()
    for face in results:
        if face["person_name"] is not None:
            continue
        emb = face.get("embedding")
        if emb is None:
            continue
        # Use a string key so it round-trips through JSON without JS precision loss
        emb_key = str(hash(emb.tobytes()[:32]))
        if emb_key in _skipped_embeddings:
            if now - _skipped_embeddings[emb_key] < UNKNOWN_RE_QUEUE_INTERVAL:
                continue

        already = any(r.get("emb_key") == emb_key for r in _reg_queue)
        if already:
            continue

        # Crop the face
        x, y, w, h = face["face_bbox"]
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            continue
        _, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(enc.tobytes()).decode("utf-8")

        _reg_queue.append({
            "emb_key": emb_key,
            "embedding": emb.tolist(),
            "face_image": b64,
            "queued_at": now,
        })

    if _reg_queue and not _reg_processing:
        _emit_next_registration()


def _emit_next_registration() -> None:
    """Emit the first item in the registration queue to the frontend."""
    global _reg_processing
    if not _reg_queue:
        _reg_processing = False
        socketio.emit("all_registered", {})
        return
    _reg_processing = True
    item = _reg_queue[0]
    socketio.emit("registration_needed", {
        "face_image": item["face_image"],
        "queue_position": 1,
        "queue_total": len(_reg_queue),
        "emb_key": str(item["emb_key"]),   # always string to JS
    })


def _emit_stats_loop() -> None:
    """Background thread: push session stats to all clients every 500 ms."""
    while True:
        time.sleep(STATS_EMIT_INTERVAL)
        if _session_tracker and _stream_active:
            stats = _session_tracker.get_session_stats()
            timeline = _session_tracker.get_emotion_timeline()[-60:]
            socketio.emit("emotion_update", {
                "stats": stats,
                "timeline": [(t, e) for t, e in timeline],
            })


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    known_faces = _registry.get_all_known_faces() if _registry else []
    return render_template("dashboard.html", known_faces=known_faces)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated webcam frames."""
    return Response(
        _frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/session_stats")
def api_session_stats():
    """Return current session statistics as JSON."""
    if not _session_tracker:
        return jsonify({"error": "Session tracker not initialised"}), 503
    return jsonify(_session_tracker.get_session_stats())


@app.route("/api/export")
def api_export():
    """Trigger CSV export and return file download."""
    if not _session_tracker:
        return jsonify({"error": "Session tracker not initialised"}), 503
    export_path = str(ROOT / "data" / f"session_{int(time.time())}.csv")
    _session_tracker.export_to_csv(export_path)
    return send_file(export_path, as_attachment=True,
                     download_name="emotion_session.csv")


@app.route("/api/export_pdf")
def api_export_pdf():
    """Trigger PDF export and return file download."""
    if not _session_tracker:
        return jsonify({"error": "Session tracker not initialised"}), 503
    pdf_path = str(ROOT / "data" / f"session_{int(time.time())}.pdf")
    face_img_dir = ROOT / "data" / "face_images"
    _session_tracker.export_to_pdf(
        pdf_path,
        model_name="MobileNetV2" if _current_model_type == "mobilenet" else "Custom CNN",
        face_images_dir=str(face_img_dir) if face_img_dir.exists() else None,
    )
    return send_file(pdf_path, as_attachment=True,
                     download_name="emotion_session.pdf")


@app.route("/api/switch_model", methods=["POST"])
def api_switch_model():
    """Switch the active (primary) emotion model at runtime.

    The other model (if trained) is automatically kept as the ensemble partner
    so accuracy benefits are preserved regardless of which model is in front.
    """
    global _current_model_type, _predictor
    data = request.get_json(force=True)
    model_type = data.get("model_type", "cnn")
    if model_type not in ("cnn", "mobilenet"):
        return jsonify({"error": "model_type must be 'cnn' or 'mobilenet'"}), 400

    model = _get_or_load_model(model_type)
    if model is None:
        return jsonify({"error": f"Model '{model_type}' not trained yet."}), 404

    other_type = "mobilenet" if model_type == "cnn" else "cnn"
    ensemble_model = _get_or_load_model(other_type)   # None if not trained yet

    if _predictor:
        _predictor.switch_model(
            model, model_type,
            ensemble_model=ensemble_model,
            ensemble_model_type=other_type if ensemble_model else None,
        )
    else:
        _predictor = EmotionPredictor(
            model, model_type, _registry,
            ensemble_model=ensemble_model,
            ensemble_model_type=other_type if ensemble_model else None,
        )
    _current_model_type = model_type
    return jsonify({
        "status": "ok",
        "model_type": model_type,
        "ensemble": ensemble_model is not None,
    })


@app.route("/api/toggle_gradcam", methods=["POST"])
def api_toggle_gradcam():
    """Toggle Grad-CAM overlay on/off."""
    global _show_gradcam
    _show_gradcam = not _show_gradcam
    return jsonify({"gradcam": _show_gradcam})


@app.route("/api/gradcam")
def api_gradcam():
    """Return the Grad-CAM heatmap for the current frame as base64 PNG."""
    with _state_lock:
        frame = _raw_frame   # use raw frame so detection is not confused by annotations
    if frame is None or not _predictor:
        return jsonify({"error": "No frame available"}), 503

    detections = _detector.detect_faces(frame) if _detector else []
    if not detections:
        return jsonify({"heatmap": None})

    det = detections[0]
    x1, y1 = max(0, det.x), max(0, det.y)
    x2, y2 = min(frame.shape[1], det.x + det.w), min(frame.shape[0], det.y + det.h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return jsonify({"heatmap": None})

    face_input = _predictor._preprocess_crop(crop)
    scores = _predictor._model.predict(np.expand_dims(face_input, 0), verbose=0)[0]
    pred_idx = int(np.argmax(scores))

    from core.gradcam import generate_heatmap, blend_heatmap_on_face
    hm = generate_heatmap(_predictor._model, face_input, pred_idx, _predictor._last_conv)
    hm_resized = cv2.resize(hm, (crop.shape[1], crop.shape[0]))
    blended = blend_heatmap_on_face(crop, hm_resized)
    _, enc = cv2.imencode(".png", blended)
    return jsonify({"heatmap": base64.b64encode(enc.tobytes()).decode("utf-8")})


@app.route("/api/known_faces")
def api_known_faces():
    """Return list of all registered people."""
    if not _registry:
        return jsonify([])
    return jsonify([
        {"person_id": pid, "name": name, "sample_count": cnt}
        for pid, name, cnt in _registry.get_all_known_faces()
    ])


@app.route("/api/delete_person/<int:person_id>", methods=["DELETE"])
def api_delete_person(person_id: int):
    """Delete a registered person by ID."""
    if not _registry:
        return jsonify({"error": "Registry not initialised"}), 503
    ok = _registry.delete_person(person_id)
    return jsonify({"deleted": ok})


# ── SocketIO events ────────────────────────────────────────────────────────────
@socketio.on("register_name")
def on_register_name(data):
    """Client submits a name for the queued face."""
    global _reg_queue, _reg_processing
    name = data.get("name", "").strip()
    emb_key = data.get("emb_key")
    if not name or not _reg_queue:
        return

    item = _reg_queue[0]
    if str(item["emb_key"]) == str(emb_key):
        emb = np.array(item["embedding"], dtype=np.float32)
        _registry.register_face(name, emb)
        # Persist the face crop so the PDF report can include a photo
        try:
            img_dir = ROOT / "data" / "face_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_bytes = base64.b64decode(item["face_image"])
            (img_dir / f"{name}.jpg").write_bytes(img_bytes)
        except Exception as exc:
            logger.warning("Could not save face image for '%s': %s", name, exc)
        _reg_queue.pop(0)
        emit("registration_complete", {"name": name, "remaining": len(_reg_queue)})
        _emit_next_registration()


@socketio.on("skip_face")
def on_skip_face(data):
    """Client skips the current registration prompt."""
    global _reg_queue
    emb_key = data.get("emb_key")
    if _reg_queue and str(_reg_queue[0]["emb_key"]) == str(emb_key):
        _skipped_embeddings[str(emb_key)] = time.time()
        _reg_queue.pop(0)
        emit("registration_complete", {"name": None, "remaining": len(_reg_queue)})
        _emit_next_registration()


@socketio.on("connect")
def on_connect():
    logger.info("Client connected: %s", request.sid)


@socketio.on("browser_frame")
def on_browser_frame(data):
    """Receive a browser-captured camera frame and send back an annotated JPEG."""
    global _stream_active, _browser_session_start

    if not _browser_frame_lock.acquire(blocking=False):
        return {"ok": False, "busy": True}

    try:
        image_data = data.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "invalid_frame"}

        _stream_active = True
        annotated = _process_dashboard_frame(frame, _browser_session_start)
        ok, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY])
        if not ok:
            return {"ok": False, "error": "encode_failed"}

        emit("processed_frame", {
            "image": base64.b64encode(enc.tobytes()).decode("utf-8"),
        })
        return {"ok": True}
    except Exception:
        logger.exception("Failed to process browser camera frame")
        return {"ok": False, "error": "processing_failed"}
    finally:
        _browser_frame_lock.release()


@socketio.on("disconnect")
def on_disconnect():
    logger.info("Client disconnected: %s", request.sid)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _init_pipeline()

    _stats_thread = threading.Thread(target=_emit_stats_loop, daemon=True)
    _stats_thread.start()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5001"))
    logger.info("EmotionAI dashboard starting at http://localhost:%s", port)
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False,
                 allow_unsafe_werkzeug=True)

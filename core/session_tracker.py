"""Rolling per-frame emotion history tracker."""

import csv
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMOTION_LABELS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]
MAX_HISTORY_FRAMES = 300


@dataclass
class FrameRecord:
    """Stores all emotion data for a single frame."""
    timestamp: float          # Unix epoch seconds
    face_id: int
    emotion_label: str
    confidence_scores: List[float]  # length-7 softmax output
    face_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    person_name: Optional[str] = None


class SessionTracker:
    """Maintains a rolling window of the last MAX_HISTORY_FRAMES frame records.

    Usage:
        tracker = SessionTracker()
        tracker.add_frame(results)          # results from predictor
        stats = tracker.get_session_stats()
        tracker.export_to_csv("session.csv")
    """

    def __init__(self, max_frames: int = MAX_HISTORY_FRAMES) -> None:
        self._history: deque = deque(maxlen=max_frames)
        self._session_start: float = time.time()
        self._total_faces_seen: int = 0

    @property
    def session_duration(self) -> float:
        """Elapsed session duration in seconds."""
        return time.time() - self._session_start

    def add_frame(self, results: List[Dict]) -> None:
        """Record emotion results for one video frame.

        Args:
            results: List of dicts per detected face. Each dict must have:
                - face_id (int)
                - emotion_label (str)
                - confidence_scores (list[float] of length 7)
                - face_bbox (tuple[int,int,int,int])
                - person_name (str | None)  [optional]
        """
        ts = time.time()
        for face in results:
            self._history.append(FrameRecord(
                timestamp=ts,
                face_id=face.get("face_id", 0),
                emotion_label=face.get("emotion_label", "Neutral"),
                confidence_scores=face.get("confidence_scores", [0.0] * 7),
                face_bbox=tuple(face.get("face_bbox", (0, 0, 0, 0))),
                person_name=face.get("person_name"),
            ))
            self._total_faces_seen += 1

    def get_emotion_timeline(self) -> List[Tuple[float, str]]:
        """Return (relative_seconds, dominant_emotion) for each record.

        Returns:
            List of (seconds_since_session_start, emotion_label) tuples.
        """
        start = self._session_start
        return [
            (rec.timestamp - start, rec.emotion_label)
            for rec in self._history
        ]

    def get_session_stats(self) -> Dict:
        """Aggregate statistics over the current history window.

        Returns:
            Dict with keys:
                dominant_emotion (str),
                distribution_percent (dict[str, float]),
                avg_confidence (float),
                total_faces_detected (int),
                duration_seconds (float).
        """
        if not self._history:
            return {
                "dominant_emotion": "N/A",
                "distribution_percent": {e: 0.0 for e in EMOTION_LABELS},
                "avg_confidence": 0.0,
                "total_faces_detected": 0,
                "duration_seconds": self.session_duration,
            }

        emotion_counts: Dict[str, int] = {e: 0 for e in EMOTION_LABELS}
        confidences: List[float] = []

        for rec in self._history:
            emotion_counts[rec.emotion_label] = emotion_counts.get(rec.emotion_label, 0) + 1
            idx = EMOTION_LABELS.index(rec.emotion_label) if rec.emotion_label in EMOTION_LABELS else 6
            confidences.append(rec.confidence_scores[idx] if len(rec.confidence_scores) > idx else 0.0)

        total = sum(emotion_counts.values()) or 1
        distribution = {e: round(c / total * 100, 2) for e, c in emotion_counts.items()}
        dominant = max(emotion_counts, key=emotion_counts.get)

        return {
            "dominant_emotion": dominant,
            "distribution_percent": distribution,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "total_faces_detected": self._total_faces_seen,
            "duration_seconds": self.session_duration,
        }

    def export_to_csv(self, path: str) -> None:
        """Write the full frame history to a CSV file.

        Args:
            path: Destination CSV file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "timestamp", "face_id", "person_name", "emotion_label",
            *[f"score_{e.lower()}" for e in EMOTION_LABELS],
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._history:
                row: Dict = {
                    "timestamp": f"{rec.timestamp:.3f}",
                    "face_id": rec.face_id,
                    "person_name": rec.person_name or "",
                    "emotion_label": rec.emotion_label,
                    "bbox_x": rec.face_bbox[0],
                    "bbox_y": rec.face_bbox[1],
                    "bbox_w": rec.face_bbox[2],
                    "bbox_h": rec.face_bbox[3],
                }
                for i, e in enumerate(EMOTION_LABELS):
                    score = rec.confidence_scores[i] if i < len(rec.confidence_scores) else 0.0
                    row[f"score_{e.lower()}"] = f"{score:.4f}"
                writer.writerow(row)
        logger.info("Session CSV exported to %s (%d records)", path, len(self._history))

    def export_person_summary_csv(self, path: str) -> None:
        """Write a per-person emotion summary CSV.

        Args:
            path: Destination CSV file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        person_stats = self.get_person_stats()
        fieldnames = ["person", "dominant_emotion", "total_frames",
                      *[f"pct_{e.lower()}" for e in EMOTION_LABELS]]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, data in sorted(person_stats.items()):
                row: Dict = {
                    "person": name,
                    "dominant_emotion": data["dominant"],
                    "total_frames": data["total_frames"],
                }
                for e in EMOTION_LABELS:
                    row[f"pct_{e.lower()}"] = f"{data['distribution'].get(e, 0):.1f}%"
                writer.writerow(row)
        logger.info("Person summary CSV exported to %s", path)

    def get_person_stats(self) -> Dict:
        """Return per-person emotion distribution grouped from history.

        Returns:
            Dict keyed by person name (or "Unknown"), each value is a dict with:
                dominant (str), distribution (dict[str, float]), total_frames (int).
        """
        from collections import defaultdict
        buckets: Dict[str, Dict] = defaultdict(lambda: {
            "counts": {e: 0 for e in EMOTION_LABELS}, "total": 0
        })
        for rec in self._history:
            key = rec.person_name or "Unknown"
            buckets[key]["counts"][rec.emotion_label] = (
                buckets[key]["counts"].get(rec.emotion_label, 0) + 1
            )
            buckets[key]["total"] += 1

        result: Dict[str, Dict] = {}
        for name, data in buckets.items():
            total = data["total"] or 1
            result[name] = {
                "dominant": max(data["counts"], key=data["counts"].get),
                "distribution": {
                    e: round(c / total * 100, 1) for e, c in data["counts"].items()
                },
                "total_frames": data["total"],
            }
        return result

    def export_to_pdf(self, path: str, model_name: str = "Custom CNN",
                      face_images_dir: Optional[str] = None) -> None:
        """Generate a PDF session report.

        Args:
            path: Destination PDF file path.
            model_name: Name of the model used.
            face_images_dir: Directory containing {name}.jpg face images.
        """
        from utils.report_generator import generate_session_pdf
        stats = self.get_session_stats()
        timeline = self.get_emotion_timeline()
        person_stats = self.get_person_stats()
        generate_session_pdf(path, stats, timeline,
                             model_name=model_name,
                             person_stats=person_stats,
                             face_images_dir=face_images_dir)
        logger.info("Session PDF exported to %s", path)

    def reset(self) -> None:
        """Clear history and restart session timer."""
        self._history.clear()
        self._session_start = time.time()
        self._total_faces_seen = 0
        logger.info("Session tracker reset.")

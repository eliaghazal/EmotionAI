"""Persistent face identity registry backed by SQLite + dlib embeddings."""

import csv
import logging
import sqlite3
import time
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "data" / "faces.db"
EMBEDDING_DIM = 128
MAX_EMBEDDINGS_PER_PERSON = 5
DEFAULT_THRESHOLD = 0.55    # Euclidean distance threshold (face_recognition default is 0.6)
# dlib embeddings are designed for Euclidean distance, NOT cosine similarity.
# Same person:       ~0.1–0.4 Euclidean distance
# Different people:  ~0.6–1.0+ Euclidean distance
# 0.55 is slightly tighter than the library default (0.6) to reduce false positives.


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two 1-D face embedding vectors."""
    return float(np.linalg.norm(
        a.astype(np.float64) - b.astype(np.float64)
    ))


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.load(BytesIO(blob))


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, embedding.astype(np.float32))
    return buf.getvalue()


class FaceRegistry:
    """Persistent store for named face embeddings.

    Stores embeddings in a SQLite database.  Multiple embeddings per person
    (up to MAX_EMBEDDINGS_PER_PERSON) improve recognition robustness.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # In-memory cache: list of (person_id, name, embedding_np)
        self._cache: List[Tuple[int, str, np.ndarray]] = []
        self._load_cache()

    # ── DB helpers ─────────────────────────────────────────────────────────────
    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self._db_path))
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    embedding BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

    def _load_cache(self) -> None:
        """Load all embeddings from DB into in-memory cache."""
        self._cache.clear()
        with self._conn() as con:
            rows = con.execute("""
                SELECT e.id, p.id AS person_id, p.name, e.embedding
                FROM embeddings e
                JOIN people p ON p.id = e.person_id
            """).fetchall()
        for row in rows:
            emb = _blob_to_embedding(row["embedding"])
            self._cache.append((row["person_id"], row["name"], emb))
        logger.info("FaceRegistry: loaded %d embeddings for %d people.",
                    len(self._cache),
                    len({pid for pid, _, _ in self._cache}))

    # ── Public API ─────────────────────────────────────────────────────────────
    def register_face(self, name: str, embedding: np.ndarray) -> int:
        """Register or update a face for the given name.

        If the person already has MAX_EMBEDDINGS_PER_PERSON embeddings, the
        oldest one is replaced to keep the sample count bounded.

        Args:
            name: Display name for the person.
            embedding: 128-d face embedding float32 array.

        Returns:
            person_id (int).
        """
        name = name.strip()
        if not name:
            raise ValueError("Name must be a non-empty string.")
        embedding = embedding.astype(np.float32)

        with self._conn() as con:
            row = con.execute("SELECT id FROM people WHERE name = ?", (name,)).fetchone()
            if row is None:
                cur = con.execute(
                    "INSERT INTO people (name, created_at) VALUES (?, ?)",
                    (name, time.time()),
                )
                person_id = cur.lastrowid
            else:
                person_id = row["id"]

            # Check current embedding count
            count = con.execute(
                "SELECT COUNT(*) AS c FROM embeddings WHERE person_id = ?",
                (person_id,),
            ).fetchone()["c"]

            if count >= MAX_EMBEDDINGS_PER_PERSON:
                # Delete oldest embedding
                oldest = con.execute(
                    "SELECT id FROM embeddings WHERE person_id = ? ORDER BY created_at ASC LIMIT 1",
                    (person_id,),
                ).fetchone()
                if oldest:
                    con.execute("DELETE FROM embeddings WHERE id = ?", (oldest["id"],))

            con.execute(
                "INSERT INTO embeddings (person_id, embedding, created_at) VALUES (?, ?, ?)",
                (person_id, _embedding_to_blob(embedding), time.time()),
            )

        self._load_cache()   # refresh in-memory cache
        logger.info("Registered face: '%s' (person_id=%d)", name, person_id)
        return person_id

    def identify_face(
        self,
        embedding: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> Tuple[Optional[str], float]:
        """Find the best matching known person for a query embedding.

        Uses Euclidean distance — the metric dlib face embeddings are designed
        for.  A *smaller* distance means a *closer* match.

        Args:
            embedding: 128-d face embedding to query.
            threshold: Maximum Euclidean distance to count as a match
                       (default 0.55; face_recognition library uses 0.6).

        Returns:
            (name, confidence) tuple.  name is None if no match within
            threshold.  confidence is in [0, 1]: 1.0 = perfect match,
            0.0 = at-threshold.
        """
        if not self._cache:
            return None, 0.0

        embedding = embedding.astype(np.float64)

        # Group by person and keep the *minimum* distance per person
        person_best: dict = {}
        for person_id, name, stored_emb in self._cache:
            dist = _euclidean_distance(embedding, stored_emb)
            if person_id not in person_best or dist < person_best[person_id][1]:
                person_best[person_id] = (name, dist)

        # Find the person with the smallest distance overall
        best_name: Optional[str] = None
        best_dist: float = float("inf")
        for person_id, (name, dist) in person_best.items():
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= threshold:
            # Map distance to confidence: 0 distance → 1.0, threshold distance → 0.0
            confidence = max(0.0, 1.0 - (best_dist / threshold))
            return best_name, float(confidence)
        return None, 0.0

    def get_all_known_faces(self) -> List[Tuple[int, str, int]]:
        """Return all registered people with their embedding sample counts.

        Returns:
            List of (person_id, name, sample_count) tuples.
        """
        with self._conn() as con:
            rows = con.execute("""
                SELECT p.id, p.name, COUNT(e.id) AS sample_count,
                       p.created_at
                FROM people p
                LEFT JOIN embeddings e ON e.person_id = p.id
                GROUP BY p.id
                ORDER BY p.name
            """).fetchall()
        return [(row["id"], row["name"], row["sample_count"]) for row in rows]

    def get_person_details(self, person_id: int) -> Optional[dict]:
        """Return full details for one person.

        Args:
            person_id: Integer primary key.

        Returns:
            Dict with name, created_at, sample_count; or None if not found.
        """
        with self._conn() as con:
            row = con.execute("""
                SELECT p.name, p.created_at, COUNT(e.id) AS sample_count
                FROM people p
                LEFT JOIN embeddings e ON e.person_id = p.id
                WHERE p.id = ?
                GROUP BY p.id
            """, (person_id,)).fetchone()
        if row is None:
            return None
        return {
            "person_id": person_id,
            "name": row["name"],
            "created_at": row["created_at"],
            "sample_count": row["sample_count"],
        }

    def delete_person(self, person_id: int) -> bool:
        """Remove a person and all their embeddings.

        Args:
            person_id: Integer primary key.

        Returns:
            True if deleted, False if person_id was not found.
        """
        with self._conn() as con:
            result = con.execute("DELETE FROM people WHERE id = ?", (person_id,))
        if result.rowcount:
            self._load_cache()
            logger.info("Deleted person id=%d from registry.", person_id)
            return True
        return False

    def export_registry(self, path: str = "data/registry_export.csv") -> None:
        """Export all known people to a CSV file.

        Args:
            path: Destination CSV file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            rows = con.execute("""
                SELECT p.id, p.name, p.created_at, COUNT(e.id) AS sample_count
                FROM people p
                LEFT JOIN embeddings e ON e.person_id = p.id
                GROUP BY p.id
                ORDER BY p.name
            """).fetchall()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "created_at", "sample_count"])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "sample_count": row["sample_count"],
                })
        logger.info("Registry exported to %s (%d people).", path, len(rows))

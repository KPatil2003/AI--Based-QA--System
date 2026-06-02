

import json
import math
from typing import Optional

from src.core.llm import get_embedding_model
from src.models.database import execute, fetch_all
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.90
MAX_CACHE_SIZE = 200


#  Math helpers 

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (no extra deps needed)."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


#  DB bootstrap 

def _ensure_cache_table() -> None:
    """Create the cache table if it doesn't exist yet."""
    execute("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            marks       INTEGER NOT NULL DEFAULT 10,
            question    TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            embedding   TEXT    NOT NULL,   -- JSON list of floats
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)


#  SemanticCache class 

class SemanticCache:
    """
    Per-user semantic cache.

    Think of it like a smart FAQ lookup:
    "Did I answer something basically the same before?
     If yes, save the LLM call and return that answer."
    """

    def __init__(
        self,
        user_id: int,
        marks: int = 10,
        threshold: float = DEFAULT_THRESHOLD
    ):
        self.user_id   = user_id
        self.marks     = marks
        self.threshold = threshold
        self._embedder = None
        _ensure_cache_table()

    #  lazy-load embedder 
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedding_model()
        return self._embedder

    def _embed(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

    # ── public API 

    def get(self, question: str) -> Optional[str]:
        """
        Return a cached answer if a semantically similar question exists,
        otherwise return None.

        Similarity check: cosine(new_embedding, cached_embedding) >= threshold
        """
        try:
            rows = fetch_all(
                """SELECT question, answer, embedding
                   FROM semantic_cache
                   WHERE user_id = ? AND marks = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (self.user_id, self.marks, MAX_CACHE_SIZE)
            )

            if not rows:
                return None

            new_emb = self._embed(question)

            best_score  = 0.0
            best_answer = None

            for row in rows:
                try:
                    cached_emb = json.loads(row["embedding"])
                    score      = _cosine_similarity(new_emb, cached_emb)
                    if score > best_score:
                        best_score  = score
                        best_answer = row["answer"]
                except Exception:
                    continue

            if best_score >= self.threshold:
                logger.info(
                    "Cache HIT for user %d (score=%.4f, threshold=%.2f): %s",
                    self.user_id, best_score, self.threshold, question[:60]
                )
                return best_answer

            logger.debug(
                "Cache MISS for user %d (best_score=%.4f): %s",
                self.user_id, best_score, question[:60]
            )
            return None

        except Exception as e:
            # Cache errors must NEVER break the main pipeline
            logger.warning("Semantic cache get() error (non-fatal): %s", e)
            return None

    def set(self, question: str, answer: str) -> None:
        """
        Store a new Q&A pair in the cache.
        Prunes oldest entries if the per-user cache is over the size limit.
        """
        try:
            embedding_json = json.dumps(self._embed(question))

            execute(
                """INSERT INTO semantic_cache
                   (user_id, marks, question, answer, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.user_id, self.marks, question, answer, embedding_json)
            )

            # Prune oldest rows if over the per-user cap
            self._prune()
            logger.debug("Cache SET for user %d: %s", self.user_id, question[:60])

        except Exception as e:
            logger.warning("Semantic cache set() error (non-fatal): %s", e)

    def invalidate(self) -> None:
        """Clear ALL cached entries for this user (e.g. after a new upload)."""
        try:
            execute(
                "DELETE FROM semantic_cache WHERE user_id = ?",
                (self.user_id,)
            )
            logger.info("Cache invalidated for user %d", self.user_id)
        except Exception as e:
            logger.warning("Cache invalidate() error: %s", e)

    #  private 

    def _prune(self) -> None:
        """Keep only the MAX_CACHE_SIZE most recent rows for this user."""
        try:
            execute(
                """DELETE FROM semantic_cache
                   WHERE user_id = ?
                   AND id NOT IN (
                       SELECT id FROM semantic_cache
                       WHERE user_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?
                   )""",
                (self.user_id, self.user_id, MAX_CACHE_SIZE)
            )
        except Exception as e:
            logger.warning("Cache prune error: %s", e)
"""
tool_registry.py — persists tools and enables semantic retrieval.

Storage design:
  SQLite  → structured metadata (name, code, stats, status, versions, feedback)
  FAISS   → vector index for cosine similarity search

Why two stores?
  SQLite  = durable, queryable, inspectable with any DB tool
  FAISS   = sub-millisecond vector search even at 10k tools

Consistency strategy:
  The FAISS index is rebuilt from SQLite on startup.  This is O(n) but
  n (number of tools) will be small (<1000) in practice, so startup cost
  is negligible.  For production at scale, persist the FAISS index to disk
  via faiss.write_index / faiss.read_index.

Tool lifecycle:
  active   → normal operation
  degraded → combined health score dropped below DEGRADED_THRESHOLD after
             MIN_EXECUTIONS_TO_DEGRADE executions; still usable but
             ranked lower in search results
  retired  → excluded from search entirely; set manually or by prune()

Health score:
  combined = EXECUTION_WEIGHT * exec_success_rate
           + SENTIMENT_WEIGHT * user_sentiment_score  (if feedback exists)
  If no user feedback exists, combined == exec_success_rate (backward compat).

Versioning:
  Every call to update_tool() snapshots the current code+description into
  tool_versions before overwriting, so the full history is preserved.

Thread safety note:
  SQLite is opened with check_same_thread=False; FAISS IndexFlatIP is NOT
  thread-safe.  Guard _rebuild_index / _add_to_index / search with
  self._index_lock if used in a multi-threaded context.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import numpy as np
import faiss

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .embeddings import embed

# ── Execution-health thresholds ───────────────────────────────────────────────
DEGRADED_THRESHOLD = 0.60       # combined score below this → "degraded"
RETIRE_THRESHOLD   = 0.35       # combined score below this (enough data) → "retired"
MIN_EXECUTIONS_TO_DEGRADE = 5   # minimum executions before degradation kicks in
MIN_EXECUTIONS_TO_RETIRE  = 10  # minimum executions before auto-retire kicks in

# ── Feedback blending weights ─────────────────────────────────────────────────
EXECUTION_WEIGHT = 0.70         # share of execution success rate in combined score
SENTIMENT_WEIGHT = 0.30         # share of user sentiment score in combined score

# ── Pruning defaults ──────────────────────────────────────────────────────────
STALE_DAYS_DEFAULT     = 30     # days without use before a tool is stale
DUPLICATE_SIMILARITY   = 0.97   # cosine similarity above which weaker tool is retired

EMBEDDING_DIM = 384


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    code: str
    usage_count: int = 0
    success_rate: float = 1.0
    status: str = "active"      # "active" | "degraded" | "retired"
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    tool_id: Optional[int] = None


@dataclass
class RetrievalResult:
    tool: Tool
    similarity: float           # cosine similarity ∈ [0, 1]


@dataclass
class ToolVersion:
    version_id: int
    tool_id: int
    version_num: int
    code: str
    description: str
    reason: str
    created_at: float


# ── Registry ──────────────────────────────────────────────────────────────────

class ToolRegistry:

    def __init__(self, db_path: str = "skills.db"):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        self._migrate_schema()

        self._index_lock = threading.Lock()
        # IndexFlatIP = exact inner-product search.
        # Unit-norm vectors → inner product == cosine similarity.
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._id_map: List[int] = []    # faiss position → tool_id in SQLite
        self._rebuild_index()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tools (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT    NOT NULL UNIQUE,
                description     TEXT    NOT NULL,
                code            TEXT    NOT NULL,
                embedding       BLOB    NOT NULL,
                usage_count     INTEGER NOT NULL DEFAULT 0,
                success_count   INTEGER NOT NULL DEFAULT 0,
                failure_count   INTEGER NOT NULL DEFAULT 0,
                status          TEXT    NOT NULL DEFAULT 'active',
                created_at      REAL    NOT NULL,
                last_used_at    REAL
            );

            CREATE TABLE IF NOT EXISTS tool_feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id     INTEGER NOT NULL REFERENCES tools(id),
                positive    INTEGER NOT NULL,
                comment     TEXT    NOT NULL DEFAULT '',
                created_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_id     INTEGER NOT NULL REFERENCES tools(id),
                version_num INTEGER NOT NULL,
                code        TEXT    NOT NULL,
                description TEXT    NOT NULL,
                reason      TEXT    NOT NULL DEFAULT '',
                created_at  REAL    NOT NULL
            );
        """)
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """
        Add columns that were introduced after the initial schema.
        SQLite does not support ALTER TABLE ADD COLUMN IF NOT EXISTS,
        so we catch OperationalError (duplicate column name) instead.
        """
        for col_sql in [
            "ALTER TABLE tools ADD COLUMN last_used_at REAL",
        ]:
            try:
                self._conn.execute(col_sql)
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    # ── FAISS index ───────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """
        Load all non-retired tools from SQLite and build the FAISS index.
        Called at startup and after any update that changes embeddings or
        retires tools.  O(n) but n is small in practice.
        """
        with self._index_lock:
            self._index.reset()
            self._id_map.clear()

            rows = self._conn.execute(
                "SELECT id, embedding FROM tools WHERE status != 'retired'"
            ).fetchall()

            if not rows:
                return

            embeddings = np.stack([
                np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
            ])
            self._index.add(embeddings)
            self._id_map.extend(row["id"] for row in rows)

    def _add_to_index(self, tool_id: int, embedding: np.ndarray) -> None:
        """Incrementally add one vector to the live FAISS index."""
        with self._index_lock:
            self._index.add(embedding.reshape(1, -1))
            self._id_map.append(tool_id)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def save_tool(self, tool: Tool, embedding: np.ndarray) -> int:
        """
        Persist a validated tool.  Returns the assigned tool_id.
        Raises sqlite3.IntegrityError if a tool with the same name exists.
        """
        emb_bytes = embedding.astype(np.float32).tobytes()
        cursor = self._conn.execute(
            """
            INSERT INTO tools (name, description, code, embedding, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tool.name, tool.description, tool.code, emb_bytes, time.time()),
        )
        self._conn.commit()
        tool_id = cursor.lastrowid
        self._add_to_index(tool_id, embedding)
        return tool_id

    def get_tool_by_id(self, tool_id: int) -> Optional[Tool]:
        row = self._conn.execute(
            "SELECT * FROM tools WHERE id = ?", (tool_id,)
        ).fetchone()
        return self._row_to_tool(row) if row else None

    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        row = self._conn.execute(
            "SELECT * FROM tools WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_tool(row) if row else None

    def list_tools(self, include_retired: bool = False) -> List[Tool]:
        if include_retired:
            rows = self._conn.execute(
                "SELECT * FROM tools ORDER BY usage_count DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tools WHERE status != 'retired' ORDER BY usage_count DESC"
            ).fetchall()
        return [self._row_to_tool(r) for r in rows]

    def update_tool(
        self,
        tool_id: int,
        new_code: str,
        new_description: str,
        new_embedding: np.ndarray,
        reason: str = "",
    ) -> None:
        """
        Replace a tool's code/description/embedding, snapshotting the current
        state as a new version first.  Resets status to 'active'.

        The FAISS index is rebuilt after the update because IndexFlatIP does
        not support in-place vector replacement.  O(n) but n is small.
        """
        with self._conn:
            # Snapshot current state
            row = self._conn.execute(
                "SELECT code, description FROM tools WHERE id = ?", (tool_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Tool id={tool_id} not found")

            version_num = self._conn.execute(
                "SELECT COALESCE(MAX(version_num), 0) + 1 FROM tool_versions WHERE tool_id = ?",
                (tool_id,),
            ).fetchone()[0]

            self._conn.execute(
                """
                INSERT INTO tool_versions (tool_id, version_num, code, description, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tool_id, version_num, row["code"], row["description"], reason, time.time()),
            )

            emb_bytes = new_embedding.astype(np.float32).tobytes()
            self._conn.execute(
                """
                UPDATE tools
                SET code = ?, description = ?, embedding = ?, status = 'active'
                WHERE id = ?
                """,
                (new_code, new_description, emb_bytes, tool_id),
            )

        self._rebuild_index()

    def get_versions(self, tool_id: int) -> List[ToolVersion]:
        """Return all historical versions of a tool, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM tool_versions WHERE tool_id = ? ORDER BY version_num ASC",
            (tool_id,),
        ).fetchall()
        return [
            ToolVersion(
                version_id=r["id"],
                tool_id=r["tool_id"],
                version_num=r["version_num"],
                code=r["code"],
                description=r["description"],
                reason=r["reason"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Semantic search over active/degraded tools.

        Returns up to top_k results sorted by:
          1. status (active > degraded)
          2. cosine similarity (descending)
        """
        with self._index_lock:
            if self._index.ntotal == 0:
                return []

            q_emb = embed(query).reshape(1, -1)
            k = min(top_k, self._index.ntotal)
            similarities, positions = self._index.search(q_emb, k)

        results: List[RetrievalResult] = []
        for sim, pos in zip(similarities[0], positions[0]):
            if pos < 0 or sim < min_similarity:
                continue
            tool_id = self._id_map[pos]
            tool = self.get_tool_by_id(tool_id)
            if tool and tool.status != "retired":
                results.append(RetrievalResult(tool=tool, similarity=float(sim)))

        # Prefer active tools over degraded; break ties by similarity
        results.sort(
            key=lambda r: (r.tool.status == "active", r.similarity),
            reverse=True,
        )
        return results

    # ── Metrics ───────────────────────────────────────────────────────────────

    def record_execution(self, tool_id: int, success: bool) -> None:
        """
        Update usage stats and re-evaluate tool health.
        Called after every tool execution, regardless of outcome.
        """
        col = "success_count" if success else "failure_count"
        self._conn.execute(
            f"UPDATE tools SET usage_count = usage_count + 1, "
            f"{col} = {col} + 1, last_used_at = ? WHERE id = ?",
            (time.time(), tool_id),
        )
        self._conn.commit()
        self._evaluate_health(tool_id)

    def _evaluate_health(self, tool_id: int) -> None:
        """
        Compute the combined health score and update tool status.

        combined = EXECUTION_WEIGHT * exec_rate + SENTIMENT_WEIGHT * sentiment
        If no feedback exists, combined == exec_rate (backward compatible).

        Thresholds:
          combined < RETIRE_THRESHOLD   and total >= MIN_EXECUTIONS_TO_RETIRE → retired
          combined < DEGRADED_THRESHOLD and total >= MIN_EXECUTIONS_TO_DEGRADE → degraded
          otherwise                                                             → active
        """
        row = self._conn.execute(
            "SELECT success_count, failure_count, status FROM tools WHERE id = ?",
            (tool_id,),
        ).fetchone()
        if not row:
            return

        total = row["success_count"] + row["failure_count"]
        if total < MIN_EXECUTIONS_TO_DEGRADE:
            return     # not enough data yet

        exec_rate = row["success_count"] / total
        sentiment = self.get_user_sentiment(tool_id)

        if sentiment is not None:
            combined = EXECUTION_WEIGHT * exec_rate + SENTIMENT_WEIGHT * sentiment
        else:
            combined = exec_rate

        # Auto-retire if deeply degraded with enough data
        if combined < RETIRE_THRESHOLD and total >= MIN_EXECUTIONS_TO_RETIRE:
            if row["status"] != "retired":
                self._retire_tool_db_only(tool_id)
                self._rebuild_index()
            return

        new_status = "degraded" if combined < DEGRADED_THRESHOLD else "active"
        if new_status != row["status"] and row["status"] != "retired":
            self._conn.execute(
                "UPDATE tools SET status = ? WHERE id = ?",
                (new_status, tool_id),
            )
            self._conn.commit()

    # ── Feedback ──────────────────────────────────────────────────────────────

    def save_feedback(self, tool_id: int, positive: bool, comment: str = "") -> None:
        """
        Record user feedback (thumbs up/down) for a tool.
        Triggers health re-evaluation so status can update immediately.
        """
        self._conn.execute(
            "INSERT INTO tool_feedback (tool_id, positive, comment, created_at) VALUES (?, ?, ?, ?)",
            (tool_id, int(positive), comment, time.time()),
        )
        self._conn.commit()
        self._evaluate_health(tool_id)

    def get_user_sentiment(self, tool_id: int) -> Optional[float]:
        """
        Returns the fraction of positive feedback votes ∈ [0, 1].
        Returns None if no feedback has been recorded for this tool.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) as total, SUM(positive) as pos FROM tool_feedback WHERE tool_id = ?",
            (tool_id,),
        ).fetchone()
        if not row or row["total"] == 0:
            return None
        return row["pos"] / row["total"]

    # ── Pruning ───────────────────────────────────────────────────────────────

    def retire_tool(self, tool_id: int) -> None:
        """Manually retire a tool.  Removes it from the FAISS index."""
        self._retire_tool_db_only(tool_id)
        self._rebuild_index()

    def _retire_tool_db_only(self, tool_id: int) -> None:
        """Update status in SQLite without rebuilding the FAISS index."""
        self._conn.execute(
            "UPDATE tools SET status = 'retired' WHERE id = ?", (tool_id,)
        )
        self._conn.commit()

    def prune(self, stale_days: int = STALE_DAYS_DEFAULT) -> List[dict]:
        """
        Retire tools matching any of three policies:

          1. Stale      — not used (or created) within stale_days days
          2. Degraded   — success_rate below RETIRE_THRESHOLD after enough executions
          3. Duplicate  — cosine similarity > DUPLICATE_SIMILARITY with another tool;
                          the one with fewer uses (or higher id on tie) is retired

        Returns a list of {"name": str, "reason": str} dicts for every tool retired
        in this call.  Rebuilds the FAISS index once at the end.
        """
        retired: List[dict] = []
        cutoff = time.time() - stale_days * 86_400

        # ── Policy 1: Stale ───────────────────────────────────────────────────
        rows = self._conn.execute(
            """
            SELECT id, name FROM tools
            WHERE status != 'retired'
            AND COALESCE(last_used_at, created_at) < ?
            """,
            (cutoff,),
        ).fetchall()
        for r in rows:
            self._retire_tool_db_only(r["id"])
            retired.append({"name": r["name"], "reason": f"Stale: not used in {stale_days} days"})

        # ── Policy 2: Deeply degraded ─────────────────────────────────────────
        rows = self._conn.execute(
            """
            SELECT id, name, success_count, failure_count FROM tools
            WHERE status != 'retired'
            AND (success_count + failure_count) >= ?
            AND CAST(success_count AS REAL) / (success_count + failure_count) < ?
            """,
            (MIN_EXECUTIONS_TO_RETIRE, RETIRE_THRESHOLD),
        ).fetchall()
        for r in rows:
            self._retire_tool_db_only(r["id"])
            retired.append({"name": r["name"], "reason": f"Deeply degraded (success rate < {RETIRE_THRESHOLD:.0%})"})

        # ── Policy 3: Duplicate detection ─────────────────────────────────────
        active_rows = self._conn.execute(
            "SELECT id, name, embedding, usage_count FROM tools WHERE status != 'retired'"
        ).fetchall()

        if len(active_rows) >= 2:
            ids        = [r["id"]          for r in active_rows]
            names      = [r["name"]        for r in active_rows]
            uses       = [r["usage_count"] for r in active_rows]
            embeddings = np.stack([
                np.frombuffer(r["embedding"], dtype=np.float32) for r in active_rows
            ])
            sims = embeddings @ embeddings.T   # (N, N) cosine similarity matrix

            already_retired_ids: set[int] = {r["name"] for r in retired}  # track by name
            already_retired_ids = set()   # reset: track by tool_id

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    if ids[i] in already_retired_ids or ids[j] in already_retired_ids:
                        continue
                    if sims[i, j] > DUPLICATE_SIMILARITY:
                        # Retire the one with fewer uses; break ties by higher id
                        weaker_idx = j if uses[i] >= uses[j] else i
                        stronger_idx = i if weaker_idx == j else j
                        self._retire_tool_db_only(ids[weaker_idx])
                        already_retired_ids.add(ids[weaker_idx])
                        retired.append({
                            "name": names[weaker_idx],
                            "reason": (
                                f"Duplicate of '{names[stronger_idx]}' "
                                f"(similarity={sims[i, j]:.3f})"
                            ),
                        })

        # Rebuild FAISS once for all retirements
        if retired:
            self._rebuild_index()

        return retired

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_tool(row: sqlite3.Row) -> Tool:
        sc = row["success_count"]
        fc = row["failure_count"]
        total = sc + fc
        rate = sc / total if total > 0 else 1.0
        return Tool(
            tool_id=row["id"],
            name=row["name"],
            description=row["description"],
            code=row["code"],
            usage_count=row["usage_count"],
            success_rate=rate,
            status=row["status"],
            created_at=row["created_at"],
            last_used_at=row["last_used_at"] if "last_used_at" in row.keys() else None,
        )

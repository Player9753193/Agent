# memory_store.py
from __future__ import annotations

import os
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _now_ts() -> float:
    return time.time()


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


@dataclass
class MemoryItem:
    id: int
    key: str
    type: str
    content: Any
    tags: List[str]
    source: str
    confidence: float
    reason: str
    note: str
    created_at: float
    updated_at: float
    deleted_at: Optional[float]


class MemoryStore:
    """
    Simple SQLite-backed memory store with:
    - key-based "current" table
    - immutable revision history
    - soft-delete
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        _ensure_dir(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    deleted_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_revisions (
                    rev_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    note TEXT NOT NULL,
                    ts REAL NOT NULL,
                    FOREIGN KEY(item_id) REFERENCES memories(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_key ON memories(key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_deleted ON memories(deleted_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rev_item ON memory_revisions(item_id)")
            conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            key=row["key"],
            type=row["type"],
            content=json.loads(row["content_json"]),
            tags=json.loads(row["tags_json"]),
            source=row["source"],
            confidence=row["confidence"],
            reason=row["reason"],
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row["deleted_at"],
        )

    def put(
        self,
        actor: str,
        key: str,
        type: str,
        content: Any,
        tags: Optional[List[str]] = None,
        source: str = "model",
        confidence: float = 0.6,
        reason: str = "",
        overwrite: bool = True,
        note: str = "",
    ) -> Dict[str, Any]:
        tags = tags or []
        now = _now_ts()
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM memories WHERE key = ? ORDER BY id DESC LIMIT 1", (key,))
            row = cur.fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO memories (key,type,content_json,tags_json,source,confidence,reason,note,created_at,updated_at,deleted_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,NULL)
                    """,
                    (key, type, json.dumps(content, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                     source, confidence, reason, note, now, now),
                )
                item_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    """
                    INSERT INTO memory_revisions (item_id,actor,content_json,tags_json,type,source,confidence,reason,note,ts)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (item_id, actor, json.dumps(content, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                     type, source, confidence, reason, note, now),
                )
                conn.commit()
                return {"ok": True, "created": True, "item_id": item_id}
            else:
                if not overwrite:
                    return {"ok": False, "error": "exists", "item_id": row["id"]}
                item_id = row["id"]
                conn.execute(
                    """
                    UPDATE memories SET type=?, content_json=?, tags_json=?, source=?, confidence=?, reason=?, note=?, updated_at=?, deleted_at=NULL
                    WHERE id=?
                    """,
                    (type, json.dumps(content, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                     source, confidence, reason, note, now, item_id),
                )
                conn.execute(
                    """
                    INSERT INTO memory_revisions (item_id,actor,content_json,tags_json,type,source,confidence,reason,note,ts)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (item_id, actor, json.dumps(content, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                     type, source, confidence, reason, note, now),
                )
                conn.commit()
                return {"ok": True, "created": False, "item_id": item_id}

    def get(self, key: str, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            if include_deleted:
                cur = conn.execute("SELECT * FROM memories WHERE key = ? ORDER BY id DESC LIMIT 1", (key,))
            else:
                cur = conn.execute("SELECT * FROM memories WHERE key = ? AND deleted_at IS NULL ORDER BY id DESC LIMIT 1", (key,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_item(row).__dict__

    def search(self, q: str = "", type: Optional[str] = None, tag: Optional[str] = None, limit: int = 20, include_deleted: bool = False) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if type:
            clauses.append("type = ?")
            params.append(type)
        if q:
            clauses.append("(key LIKE ? OR content_json LIKE ? OR reason LIKE ? OR note LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like, like])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        out: List[Dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                item = self._row_to_item(r).__dict__
                if tag:
                    if tag in item.get("tags", []):
                        out.append(item)
                else:
                    out.append(item)
        return out

    def update(
        self,
        actor: str,
        item_id: int,
        content: Any,
        tags: Optional[List[str]] = None,
        type: Optional[str] = None,
        source: str = "model",
        confidence: float = 0.6,
        reason: str = "",
        note: str = "",
    ) -> Dict[str, Any]:
        tags = tags or []
        now = _now_ts()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                return {"ok": False, "error": "not_found"}
            cur_type = type or row["type"]
            conn.execute(
                """
                UPDATE memories SET type=?, content_json=?, tags_json=?, source=?, confidence=?, reason=?, note=?, updated_at=?
                WHERE id=?
                """,
                (cur_type, json.dumps(content, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                 source, confidence, reason, note, now, item_id),
            )
            conn.execute(
                """
                INSERT INTO memory_revisions (item_id,actor,content_json,tags_json,type,source,confidence,reason,note,ts)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (item_id, actor, json.dumps(content, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                 cur_type, source, confidence, reason, note, now),
            )
            conn.commit()
            return {"ok": True, "item_id": item_id}

    def delete(self, actor: str, item_id: int, reason: str = "") -> Dict[str, Any]:
        now = _now_ts()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                return {"ok": False, "error": "not_found"}
            conn.execute("UPDATE memories SET deleted_at=?, reason=? , updated_at=? WHERE id=?",
                         (now, reason or row["reason"], now, item_id))
            conn.execute(
                """
                INSERT INTO memory_revisions (item_id,actor,content_json,tags_json,type,source,confidence,reason,note,ts)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (item_id, actor, row["content_json"], row["tags_json"], row["type"], row["source"], row["confidence"],
                 reason or row["reason"], row["note"], now),
            )
            conn.commit()
            return {"ok": True, "deleted": True, "item_id": item_id}

    def history(self, item_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_revisions WHERE item_id=? ORDER BY ts DESC LIMIT ?",
                (item_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def revert(self, actor: str, item_id: int, rev_id: int, reason: str = "") -> Dict[str, Any]:
        with self._connect() as conn:
            rev = conn.execute("SELECT * FROM memory_revisions WHERE rev_id=? AND item_id=?",
                               (rev_id, item_id)).fetchone()
            if rev is None:
                return {"ok": False, "error": "rev_not_found"}

            now = _now_ts()
            conn.execute(
                """
                UPDATE memories SET type=?, content_json=?, tags_json=?, source=?, confidence=?, reason=?, note=?, updated_at=?
                WHERE id=?
                """,
                (rev["type"], rev["content_json"], rev["tags_json"], rev["source"], rev["confidence"],
                 reason or rev["reason"], rev["note"], now, item_id),
            )
            conn.execute(
                """
                INSERT INTO memory_revisions (item_id,actor,content_json,tags_json,type,source,confidence,reason,note,ts)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (item_id, actor, rev["content_json"], rev["tags_json"], rev["type"], rev["source"], rev["confidence"],
                 reason or rev["reason"], rev["note"], now),
            )
            conn.commit()
            return {"ok": True, "item_id": item_id, "reverted_to_rev": rev_id}

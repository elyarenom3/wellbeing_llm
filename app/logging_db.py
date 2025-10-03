from __future__ import annotations
import sqlite3, json, os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from . import privacy

DEFAULT_PATH = os.environ.get("WB_SQLITE_PATH", "wellbeing_logs.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY,
  user_id TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS steps(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  step_name TEXT,
  input_json TEXT,
  output_json TEXT,
  started_at TEXT,
  ended_at TEXT,
  meta TEXT
);
CREATE TABLE IF NOT EXISTS plans(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  plan_json TEXT,
  signals_json TEXT,
  created_at TEXT
);
CREATE TABLE IF NOT EXISTS user_metrics(
  user_id TEXT PRIMARY KEY,
  streak INTEGER DEFAULT 0,
  total_sessions INTEGER DEFAULT 0,
  last_seen TEXT,
  last_reflection_score REAL,
  rolling_reflection_score REAL,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS life_quality(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  user_id TEXT,
  score REAL,
  payload TEXT,
  created_at TEXT
);
"""

def get_conn(db_path: Optional[str] = None):
  path = db_path or DEFAULT_PATH
  conn = sqlite3.connect(path)
  return conn

def init_db(db_path: Optional[str] = None):
  conn = get_conn(db_path)
  with conn:
    conn.executescript(SCHEMA)
  conn.close()

def create_session(session_id: str, user_id: str, db_path: Optional[str] = None, created_at: Optional[str] = None):
  if not privacy.should_log():
    return
  conn = get_conn(db_path)
  with conn:
    conn.execute("INSERT OR REPLACE INTO sessions(id, user_id, created_at) VALUES (?, ?, ?)", 
                 (session_id, user_id, created_at or datetime.utcnow().isoformat()))
  conn.close()

def log_step(session_id: str, step_name: str, input_obj: Dict[str, Any], output_obj: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, db_path: Optional[str] = None):
  if not privacy.should_log():
    return
  prepared = privacy.prepare_step_storage(step_name, input_obj, output_obj)
  conn = get_conn(db_path)
  with conn:
    conn.execute(
      "INSERT INTO steps(session_id, step_name, input_json, output_json, started_at, ended_at, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (
        session_id,
        step_name,
        prepared["input"],
        prepared["output"],
        datetime.utcnow().isoformat(),
        datetime.utcnow().isoformat(),
        privacy.encrypt_payload(meta or {}),
      )
    )
  conn.close()

def save_plan(session_id: str, plan_obj: Dict[str, Any], signals_obj: Dict[str, Any], db_path: Optional[str] = None):
  if not privacy.should_log():
    return
  prepared = privacy.prepare_plan_storage(plan_obj, signals_obj)
  conn = get_conn(db_path)
  with conn:
    conn.execute(
      "INSERT INTO plans(session_id, plan_json, signals_json, created_at) VALUES (?, ?, ?, ?)",
      (
        session_id,
        prepared["plan"],
        prepared["signals"],
        datetime.utcnow().isoformat(),
      )
    )
  conn.close()


def record_life_quality(session_id: str, user_id: str, score: float, payload: Dict[str, Any], db_path: Optional[str] = None) -> None:
  if not privacy.should_log():
    return
  conn = get_conn(db_path)
  with conn:
    conn.execute(
      "INSERT INTO life_quality(session_id, user_id, score, payload, created_at) VALUES (?, ?, ?, ?, ?)",
      (
        session_id,
        user_id,
        float(score),
        privacy.encrypt_payload(payload),
        datetime.utcnow().isoformat(),
      )
    )
  conn.close()


def fetch_life_quality_history(user_id: str, limit: int = 7, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
  conn = get_conn(db_path)
  cur = conn.cursor()
  cur.execute(
    "SELECT score, created_at FROM life_quality WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
    (user_id, limit),
  )
  rows = cur.fetchall()
  conn.close()
  return [{"score": r[0], "created_at": r[1]} for r in reversed(rows)]


def get_user_metrics(user_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
  conn = get_conn(db_path)
  cur = conn.cursor()
  cur.execute(
    "SELECT user_id, streak, total_sessions, last_seen, last_reflection_score, rolling_reflection_score, updated_at FROM user_metrics WHERE user_id=?",
    (user_id,),
  )
  row = cur.fetchone()
  conn.close()
  if not row:
    return None
  return {
    "user_id": row[0],
    "streak": row[1],
    "total_sessions": row[2],
    "last_seen": row[3],
    "last_reflection_score": row[4],
    "rolling_reflection_score": row[5],
    "updated_at": row[6],
  }


def upsert_user_metrics(
  user_id: str,
  streak: int,
  total_sessions: int,
  last_seen: str,
  last_reflection_score: float,
  rolling_reflection_score: float,
  db_path: Optional[str] = None,
) -> None:
  conn = get_conn(db_path)
  with conn:
    conn.execute(
      """
      INSERT INTO user_metrics(user_id, streak, total_sessions, last_seen, last_reflection_score, rolling_reflection_score, updated_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(user_id) DO UPDATE SET
        streak=excluded.streak,
        total_sessions=excluded.total_sessions,
        last_seen=excluded.last_seen,
        last_reflection_score=excluded.last_reflection_score,
        rolling_reflection_score=excluded.rolling_reflection_score,
        updated_at=excluded.updated_at
      """,
      (
        user_id,
        streak,
        total_sessions,
        last_seen,
        last_reflection_score,
        rolling_reflection_score,
        datetime.utcnow().isoformat(),
      ),
    )
  conn.close()

def fetch_steps(session_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
  conn = get_conn(db_path)
  cur = conn.cursor()
  cur.execute("SELECT step_name, input_json, output_json, started_at, ended_at, meta FROM steps WHERE session_id=? ORDER BY id ASC", (session_id,))
  rows = cur.fetchall()
  conn.close()
  results = []
  for r in rows:
    results.append({
      "step_name": r[0],
      "input": r[1],
      "output": r[2],
      "started_at": r[3],
      "ended_at": r[4],
      "meta": r[5],
    })
  return results


def run_retention_cleanup(days: int = 30, db_path: Optional[str] = None) -> None:
  if days <= 0:
    return
  cutoff = datetime.utcnow() - timedelta(days=days)
  conn = get_conn(db_path)
  with conn:
    conn.execute("DELETE FROM steps WHERE started_at < ?", (cutoff.isoformat(),))
    conn.execute("DELETE FROM plans WHERE created_at < ?", (cutoff.isoformat(),))
    conn.execute("DELETE FROM life_quality WHERE created_at < ?", (cutoff.isoformat(),))
  conn.close()

from __future__ import annotations
import sqlite3, json, os
from datetime import datetime
from typing import Dict, Any, Optional, List

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

def create_session(session_id: str, user_id: str, db_path: Optional[str] = None):
  conn = get_conn(db_path)
  with conn:
    conn.execute("INSERT OR REPLACE INTO sessions(id, user_id, created_at) VALUES (?, ?, ?)", 
                 (session_id, user_id, datetime.utcnow().isoformat()))
  conn.close()

def log_step(session_id: str, step_name: str, input_obj: Dict[str, Any], output_obj: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, db_path: Optional[str] = None):
  conn = get_conn(db_path)
  with conn:
    conn.execute(
      "INSERT INTO steps(session_id, step_name, input_json, output_json, started_at, ended_at, meta) VALUES (?, ?, ?, ?, ?, ?, ?)",
      (
        session_id,
        step_name,
        json.dumps(input_obj, ensure_ascii=False),
        json.dumps(output_obj, ensure_ascii=False),
        datetime.utcnow().isoformat(),
        datetime.utcnow().isoformat(),
        json.dumps(meta or {}, ensure_ascii=False),
      )
    )
  conn.close()

def save_plan(session_id: str, plan_obj: Dict[str, Any], signals_obj: Dict[str, Any], db_path: Optional[str] = None):
  conn = get_conn(db_path)
  with conn:
    conn.execute(
      "INSERT INTO plans(session_id, plan_json, signals_json, created_at) VALUES (?, ?, ?, ?)",
      (
        session_id,
        json.dumps(plan_obj, ensure_ascii=False),
        json.dumps(signals_obj, ensure_ascii=False),
        datetime.utcnow().isoformat(),
      )
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

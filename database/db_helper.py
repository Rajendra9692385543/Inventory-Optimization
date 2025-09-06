# database/db_helper.py
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

DB_PATH = Path(__file__).resolve().parent / "inventory.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name TEXT UNIQUE,
        sku TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_map (
        faiss_id INTEGER PRIMARY KEY,
        item_id INTEGER,
        image_path TEXT,
        added_at TEXT,
        FOREIGN KEY(item_id) REFERENCES items(id)
    )
    """)
    conn.commit()
    conn.close()

def add_item_return_id(item_name: str, sku: str, created_at: str) -> Optional[int]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO items (item_name, sku, created_at) VALUES (?, ?, ?)",
                (item_name, sku, created_at))
    conn.commit()
    cur.execute("SELECT id FROM items WHERE item_name=?", (item_name,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def add_embedding_map(faiss_id: int, item_id: int, image_path: str, added_at: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO embeddings_map (faiss_id, item_id, image_path, added_at) VALUES (?, ?, ?, ?)",
                (faiss_id, item_id, image_path, added_at))
    conn.commit()
    conn.close()

def get_item_by_id(item_id: int) -> Optional[Tuple]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM items WHERE id=?", (item_id,))
    row = cur.fetchone()
    conn.close()
    return row

def get_items_by_faiss_ids(faiss_ids: List[int]) -> List[Tuple]:
    if not faiss_ids:
        return []
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" for _ in faiss_ids)
    cur.execute(f"SELECT faiss_id, item_id, image_path FROM embeddings_map WHERE faiss_id IN ({placeholders})", tuple(faiss_ids))
    rows = cur.fetchall()
    conn.close()
    return rows

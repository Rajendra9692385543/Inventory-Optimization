from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import sqlite3
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR / "inventory.db"

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- DB SETUP ----------
@app.on_event("startup")
def startup():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_no TEXT,
            units INTEGER,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# ---------- MODELS ----------
class ProductEntry(BaseModel):
    model_no: str
    units: int

# ---------- API ROUTES ----------
@app.post("/add_product")
def add_product(entry: ProductEntry):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO inventory (model_no, units) VALUES (?, ?)",
                (entry.model_no, entry.units))
    conn.commit()
    conn.close()
    return {"status": "added", "model_no": entry.model_no, "units": entry.units}

@app.post("/reduce_stock")
def reduce_stock(entry: ProductEntry):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT units FROM inventory WHERE model_no=? ORDER BY id DESC LIMIT 1",
                (entry.model_no,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Model not found")
    current_units = row[0]
    if entry.units > current_units:
        conn.close()
        raise HTTPException(status_code=400, detail="Not enough stock")
    new_units = current_units - entry.units
    cur.execute("INSERT INTO inventory (model_no, units) VALUES (?, ?)",
                (entry.model_no, new_units))
    conn.commit()
    conn.close()
    return {"status": "reduced", "model_no": entry.model_no, "new_units": new_units}

@app.get("/inventory")
def inventory():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT model_no, units
        FROM (
            SELECT model_no, units,
                   ROW_NUMBER() OVER (PARTITION BY model_no ORDER BY ts DESC) as rn
            FROM inventory
        ) WHERE rn = 1
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"model_no": r[0], "units": r[1]} for r in rows]

# ---------- STATIC FILE ROUTES ----------
@app.get("/scanner")
def serve_scanner():
    return FileResponse(STATIC_DIR / "scanner.html")

@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

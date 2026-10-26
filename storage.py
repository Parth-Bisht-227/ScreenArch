from pathlib import Path
from datetime import datetime, timezone
import hashlib
import sqlite3

from PIL import Image

DB_PATH = Path(__file__).parent / "data" / "screenshots.db"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            modified_at REAL NOT NULL,
            content_hash TEXT,
            width INTEGER,
            height INTEGER,
            ocr_text TEXT NOT NULL DEFAULT '',
            ocr_status TEXT NOT NULL DEFAULT 'pending',
            indexed_at TEXT NOT NULL
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_screenshots_path ON screenshots(path)")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS screenshot_fts USING fts5(path UNINDEXED, ocr_text, content='')")

def _hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _ocr(path: Path):
    try:
        import pytesseract
        from PIL import ImageOps
    except ImportError:
        with Image.open(path) as image:
            return "", image.width, image.height, "ocr-unavailable"

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            try:
                text = pytesseract.image_to_string(image).strip()
            except pytesseract.pytesseract.TesseractNotFoundError:
                return "", image.width, image.height, "tesseract-missing"
            return text, image.width, image.height, "complete"
    except Exception as exc:
        try:
            with Image.open(path) as image:
                return "", image.width, image.height, f"ocr-error: {type(exc).__name__}"
        except Exception:
            return "", None, None, "image-error"

def _new_counts():
    return {"new": 0, "updated": 0, "skipped": 0, "tesseract_missing": 0, "errors": []}

def _index_path(path: Path, conn, counts):
    try:
        stat = path.stat()
        resolved_path = str(path.resolve())
        existing = conn.execute(
            "SELECT id, file_size, modified_at, ocr_status FROM screenshots WHERE path = ?",
            (resolved_path,)
        ).fetchone()
        unchanged = (
            existing
            and existing["file_size"] == stat.st_size
            and existing["modified_at"] == stat.st_mtime
        )
        if unchanged and existing["ocr_status"] == "complete":
            counts["skipped"] += 1
            return

        text, width, height, status = _ocr(path)
        content_hash = _hash_file(path)
        indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.execute("""
            INSERT INTO screenshots(path, file_size, modified_at, content_hash, width, height, ocr_text, ocr_status, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              file_size=excluded.file_size, modified_at=excluded.modified_at,
              content_hash=excluded.content_hash, width=excluded.width, height=excluded.height,
              ocr_text=excluded.ocr_text, ocr_status=excluded.ocr_status, indexed_at=excluded.indexed_at
        """, (resolved_path, stat.st_size, stat.st_mtime, content_hash, width, height, text, status, indexed_at))
        conn.execute("DELETE FROM screenshot_fts WHERE path = ?", (resolved_path,))
        conn.execute("INSERT INTO screenshot_fts(path, ocr_text) VALUES (?, ?)", (resolved_path, text))
        if status == "tesseract-missing":
            counts["tesseract_missing"] += 1
        counts["new" if not existing else "updated"] += 1
    except Exception as exc:
        counts["errors"].append(f"{path}: {type(exc).__name__}: {exc}")

def index_folder(folder: Path):
    counts = _new_counts()
    folder = folder.resolve()
    with connect() as conn:
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                _index_path(path, conn, counts)
    return counts

def retry_failed_ocr():
    counts = _new_counts()
    with connect() as conn:
        paths = conn.execute(
            "SELECT path FROM screenshots WHERE ocr_status != 'complete' ORDER BY indexed_at"
        ).fetchall()
        for row in paths:
            _index_path(Path(row["path"]), conn, counts)
    return counts

def _dict(row):
    return dict(row) if row else None

def list_items(limit=500):
    with connect() as conn:
        rows = conn.execute("SELECT * FROM screenshots ORDER BY indexed_at DESC LIMIT ?", (limit,)).fetchall()
    return [_dict(r) for r in rows]

def search_items(query, limit=500):
    # Use substring search for predictable behavior with punctuation and OCR text.
    # Semantic search will be a separate optional layer.
    q = f"%{query.strip()}%"
    with connect() as conn:
        rows = conn.execute("""
            SELECT * FROM screenshots
            WHERE path LIKE ? OR ocr_text LIKE ?
            ORDER BY indexed_at DESC LIMIT ?
        """, (q, q, limit)).fetchall()
    return [_dict(r) for r in rows]

def get_item(item_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM screenshots WHERE id = ?", (item_id,)).fetchone()
    return _dict(row)

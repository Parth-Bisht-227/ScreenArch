from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import sqlite3

from PIL import Image

DB_PATH = Path(__file__).parent / "data" / "screenshots.db"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

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
        conn.execute("""
        CREATE TABLE IF NOT EXISTS screenshot_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            screenshot_id INTEGER NOT NULL,
            prompt_hash TEXT NOT NULL,
            prompt TEXT NOT NULL,
            model TEXT NOT NULL,
            input_hash TEXT NOT NULL,
            category TEXT NOT NULL,
            reason TEXT NOT NULL,
            evidence TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            needs_review INTEGER NOT NULL CHECK(needs_review IN (0, 1)),
            result_status TEXT NOT NULL DEFAULT 'legacy',
            model_duration_seconds REAL NOT NULL DEFAULT 0,
            model_request_durations TEXT NOT NULL DEFAULT '[]',
            approved INTEGER NOT NULL DEFAULT 0 CHECK(approved IN (0, 1)),
            approved_category TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(screenshot_id, prompt_hash, model, input_hash)
        )
        """)
        classification_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(screenshot_classifications)")
        }
        for name, definition in (
            ("result_status", "TEXT NOT NULL DEFAULT 'legacy'"),
            ("model_duration_seconds", "REAL NOT NULL DEFAULT 0"),
            ("model_request_durations", "TEXT NOT NULL DEFAULT '[]'"),
            ("approved", "INTEGER NOT NULL DEFAULT 0"),
            ("approved_category", "TEXT"),
        ):
            if name not in classification_columns:
                conn.execute(
                    f"ALTER TABLE screenshot_classifications ADD COLUMN {name} {definition}"
                )
        conn.execute("""
            UPDATE screenshot_classifications
            SET result_status = CASE WHEN needs_review = 1 THEN 'needs_review' ELSE 'classified' END
            WHERE result_status = 'legacy'
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_classifications_prompt_model
        ON screenshot_classifications(prompt_hash, model)
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS organization_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            classification_id INTEGER NOT NULL,
            source_path TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            destination_path TEXT NOT NULL,
            operation TEXT NOT NULL CHECK(operation IN ('copy', 'move')),
            file_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            UNIQUE(classification_id, operation, output_dir)
        )
        """)

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

def list_classification_candidates():
    with connect() as conn:
        rows = conn.execute("SELECT * FROM screenshots ORDER BY indexed_at DESC").fetchall()
    return [_dict(row) for row in rows]

def get_saved_classification(item_id, prompt_hash, model, input_hash):
    with connect() as conn:
        row = conn.execute("""
            SELECT * FROM screenshot_classifications
            WHERE screenshot_id = ? AND prompt_hash = ? AND model = ? AND input_hash = ?
        """, (item_id, prompt_hash, model, input_hash)).fetchone()
    return _dict(row)

def save_classification(item_id, prompt_hash, prompt, model, input_hash, result):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute("""
            INSERT INTO screenshot_classifications(
                screenshot_id, prompt_hash, prompt, model, input_hash, category,
                reason, evidence, confidence, needs_review, result_status,
                model_duration_seconds, model_request_durations, approved,
                approved_category, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)
            ON CONFLICT(screenshot_id, prompt_hash, model, input_hash) DO UPDATE SET
                prompt=excluded.prompt,
                category=excluded.category,
                reason=excluded.reason,
                evidence=excluded.evidence,
                confidence=excluded.confidence,
                needs_review=excluded.needs_review,
                result_status=excluded.result_status,
                model_duration_seconds=excluded.model_duration_seconds,
                model_request_durations=excluded.model_request_durations,
                approved=0,
                approved_category=NULL,
                created_at=excluded.created_at
        """, (
            item_id, prompt_hash, prompt, model, input_hash, result["category"],
            result["reason"], result["evidence"], result["confidence"],
            int(result["needs_review"]), result.get("status", "needs_review"),
            result.get("model_duration_seconds", 0.0),
            json.dumps(result.get("model_request_durations", [])),
            created_at
        ))

def list_classifications(prompt_hash, model):
    with connect() as conn:
        rows = conn.execute("""
            SELECT c.*, s.path, s.file_size, s.modified_at, s.width, s.height,
                   s.ocr_text, s.ocr_status
            FROM screenshot_classifications AS c
            JOIN screenshots AS s ON s.id = c.screenshot_id
            WHERE c.prompt_hash = ? AND c.model = ?
            ORDER BY c.created_at DESC
        """, (prompt_hash, model)).fetchall()
    return [_dict(row) for row in rows]

def save_review_decision(classification_id, category, approved):
    category = category.strip()
    if not category:
        raise ValueError("Choose a category before saving the review decision.")
    with connect() as conn:
        cursor = conn.execute("""
            UPDATE screenshot_classifications
            SET approved_category = ?, approved = ?
            WHERE id = ?
        """, (category, int(bool(approved)), classification_id))
        if cursor.rowcount != 1:
            raise ValueError(f"Classification {classification_id} no longer exists.")

def approve_suitable_classifications(classification_ids):
    ids = list(dict.fromkeys(classification_ids))
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    with connect() as conn:
        cursor = conn.execute(f"""
            UPDATE screenshot_classifications
            SET approved = 1
            WHERE id IN ({placeholders})
              AND result_status = 'classified'
              AND needs_review = 0
              AND lower(COALESCE(approved_category, category)) != 'needs review'
        """, ids)
        return cursor.rowcount

def get_approved_classifications():
    with connect() as conn:
        rows = conn.execute("""
            SELECT c.*, s.path, s.file_size AS source_file_size,
                   s.modified_at AS source_modified_at,
                   s.content_hash AS source_content_hash
            FROM screenshot_classifications AS c
            JOIN screenshots AS s ON s.id = c.screenshot_id
            WHERE c.approved = 1
            ORDER BY c.id
        """).fetchall()
    results = [_dict(row) for row in rows]
    for row in results:
        row["category"] = row["approved_category"] or row["category"]
        row["needs_review"] = bool(row["needs_review"])
    return results

def get_organization_operation(classification_id, operation, output_dir):
    with connect() as conn:
        row = conn.execute("""
            SELECT * FROM organization_operations
            WHERE classification_id = ? AND operation = ? AND output_dir = ?
        """, (classification_id, operation, output_dir)).fetchone()
    return _dict(row)

def save_organization_operation(
    classification_id,
    source_path,
    output_dir,
    destination_path,
    operation,
    file_hash,
    status,
    error="",
):
    updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute("""
            INSERT INTO organization_operations(
                classification_id, source_path, output_dir, destination_path,
                operation, file_hash, status, error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(classification_id, operation, output_dir) DO UPDATE SET
                source_path=excluded.source_path,
                destination_path=excluded.destination_path,
                file_hash=excluded.file_hash,
                status=excluded.status,
                error=excluded.error,
                updated_at=excluded.updated_at
        """, (
            classification_id, source_path, output_dir, destination_path,
            operation, file_hash, status, error, updated_at
        ))

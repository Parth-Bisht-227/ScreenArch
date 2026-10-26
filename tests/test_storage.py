from pathlib import Path
import sys
from types import SimpleNamespace
from PIL import Image
import storage

def test_index_and_keyword_search(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()

    # Avoid requiring the system Tesseract binary in the unit test.
    monkeypatch.setattr(storage, "_ocr", lambda path: ("sample payment receipt", 40, 30, "complete"))
    monkeypatch.setattr(storage, "_hash_file", lambda path: "testhash")

    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (40, 30), "white").save(image_path)

    result = storage.index_folder(tmp_path)
    assert result["new"] == 1
    assert storage.search_items("payment")[0]["ocr_text"] == "sample payment receipt"

def test_failed_ocr_is_retried_without_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    image_path = tmp_path / "retry.png"
    Image.new("RGB", (40, 30), "white").save(image_path)
    results = iter([
        ("", 40, 30, "ocr-error: RuntimeError"),
        ("retried text", 40, 30, "complete"),
    ])
    monkeypatch.setattr(storage, "_ocr", lambda path: next(results))
    monkeypatch.setattr(storage, "_hash_file", lambda path: "testhash")

    first_result = storage.index_folder(tmp_path)
    retry_result = storage.retry_failed_ocr()

    item = storage.list_items()[0]
    assert first_result["new"] == 1
    assert retry_result["updated"] == 1
    assert item["ocr_status"] == "complete"
    assert item["ocr_text"] == "retried text"

def test_folder_rescan_retries_failed_ocr_without_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    image_path = tmp_path / "rescan.png"
    Image.new("RGB", (40, 30), "white").save(image_path)
    results = iter([
        ("", 40, 30, "ocr-error: RuntimeError"),
        ("retried on scan", 40, 30, "complete"),
    ])
    monkeypatch.setattr(storage, "_ocr", lambda path: next(results))
    monkeypatch.setattr(storage, "_hash_file", lambda path: "testhash")

    first_result = storage.index_folder(tmp_path)
    second_result = storage.index_folder(tmp_path)

    item = storage.list_items()[0]
    assert first_result["new"] == 1
    assert second_result["updated"] == 1
    assert second_result["skipped"] == 0
    assert item["ocr_status"] == "complete"
    assert item["ocr_text"] == "retried on scan"

def test_unchanged_successful_ocr_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    image_path = tmp_path / "unchanged.png"
    Image.new("RGB", (40, 30), "white").save(image_path)
    ocr_calls = []

    def successful_ocr(path):
        ocr_calls.append(path)
        return "sample text", 40, 30, "complete"

    monkeypatch.setattr(storage, "_ocr", successful_ocr)
    monkeypatch.setattr(storage, "_hash_file", lambda path: "testhash")

    first_result = storage.index_folder(tmp_path)
    second_result = storage.index_folder(tmp_path)

    assert first_result["new"] == 1
    assert second_result["skipped"] == 1
    assert len(ocr_calls) == 1

def test_missing_tesseract_is_reported_separately(tmp_path, monkeypatch):
    class TesseractNotFoundError(RuntimeError):
        pass

    fake_pytesseract = SimpleNamespace(
        image_to_string=lambda image: (_ for _ in ()).throw(TesseractNotFoundError()),
        pytesseract=SimpleNamespace(TesseractNotFoundError=TesseractNotFoundError),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    image_path = tmp_path / "no-tesseract.png"
    Image.new("RGB", (40, 30), "white").save(image_path)

    text, width, height, status = storage._ocr(image_path)

    assert (text, width, height, status) == ("", 40, 30, "tesseract-missing")

def test_no_match(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    assert storage.search_items("not found") == []

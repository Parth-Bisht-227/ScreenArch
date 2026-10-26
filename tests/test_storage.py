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


def test_folder_scoped_listing_and_keyword_search_preserve_both_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder_a = tmp_path / "folder-a"
    folder_b = tmp_path / "folder-b"
    folder_a.mkdir()
    folder_b.mkdir()

    def add_images(folder, prefix, count):
        for index in range(count):
            Image.new("RGB", (20, 20), "white").save(folder / f"{prefix}-{index}.png")

    add_images(folder_a, "a", 3)
    add_images(folder_b, "b", 7)
    monkeypatch.setattr(
        storage,
        "_ocr",
        lambda path: (f"keyword {path.parent.name}", 20, 20, "complete"),
    )

    assert storage.index_folder(folder_a)["new"] == 3
    assert storage.index_folder(folder_b)["new"] == 7

    items_b = storage.list_items(folder=folder_b)
    assert len(items_b) == 7
    assert all(storage._path_is_within(item["path"], folder_b) for item in items_b)
    assert storage.search_items("folder-a", folder=folder_b) == []
    assert len(storage.search_items("folder-b", folder=folder_b)) == 7
    assert len(storage.list_items(folder=folder_a)) == 3
    assert len(storage.search_items("folder-a", folder=folder_a)) == 3
    assert len(storage.list_items()) == 10

    alternate_path = str(folder_b).replace("\\", "/").upper()
    assert len(storage.list_items(folder=alternate_path)) == 7


def test_index_reports_removed_files_without_deleting_cached_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder = tmp_path / "screenshots"
    folder.mkdir()
    paths = [folder / f"screen-{index}.png" for index in range(3)]
    for path in paths:
        Image.new("RGB", (20, 20), "white").save(path)
    monkeypatch.setattr(
        storage, "_ocr", lambda path: ("sample OCR text", 20, 20, "complete")
    )

    assert storage.index_folder(folder)["new"] == 3
    paths[0].unlink()

    result = storage.index_folder(folder)

    assert result["removed"] == 1
    assert len(storage.list_items(folder=folder)) == 2
    assert storage.search_items(paths[0].name, folder=folder) == []
    with storage.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 3
        assert connection.execute(
            "SELECT is_removed FROM screenshots WHERE path = ?",
            (str(paths[0].resolve()),),
        ).fetchone()[0] == 1


def test_index_distinguishes_new_changed_and_unchanged_supported_images(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder = tmp_path / "screenshots"
    folder.mkdir()
    first = folder / "first.png"
    Image.new("RGB", (20, 20), "white").save(first)
    (folder / "ignored.gif").write_bytes(b"not indexed")
    ocr_calls = []
    monkeypatch.setattr(
        storage,
        "_ocr",
        lambda path: ocr_calls.append(path) or ("sample OCR text", 20, 20, "complete"),
    )

    first_scan = storage.index_folder(folder)
    second_scan = storage.index_folder(folder)
    Image.new("RGB", (40, 20), "black").save(first)
    changed_scan = storage.index_folder(folder)

    assert first_scan["new"] == 1
    assert second_scan["skipped"] == 1
    assert changed_scan["updated"] == 1
    assert len(ocr_calls) == 2
    assert len(storage.list_items(folder=folder)) == 1


def test_generated_organization_directory_is_not_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder = tmp_path / "screenshots"
    organized = folder / "ScreenArch_Organized" / "Work"
    organized.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(folder / "source.png")
    Image.new("RGB", (20, 20), "white").save(organized / "sorted.png")
    ocr_calls = []
    monkeypatch.setattr(
        storage,
        "_ocr",
        lambda path: ocr_calls.append(path) or ("sample OCR text", 20, 20, "complete"),
    )

    result = storage.index_folder(folder)

    assert result["new"] == 1
    assert [path.name for path in ocr_calls] == ["source.png"]
    assert [Path(item["path"]).name for item in storage.list_items(folder=folder)] == [
        "source.png"
    ]

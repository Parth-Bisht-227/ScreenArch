from pathlib import Path
import os

from PIL import Image
from streamlit.testing.v1 import AppTest

import classification
import ollama_client
import storage


def test_gallery_and_search_follow_selected_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder_a = tmp_path / "folder-a"
    folder_b = tmp_path / "folder-b"
    folder_a.mkdir()
    folder_b.mkdir()
    monkeypatch.setattr(
        storage,
        "_ocr",
        lambda path: (f"keyword {path.parent.name}", 20, 20, "complete"),
    )

    for folder, prefix, count in ((folder_a, "a", 3), (folder_b, "b", 7)):
        for index in range(count):
            Image.new("RGB", (20, 20), "white").save(folder / f"{prefix}-{index}.png")
        storage.index_folder(folder)

    at = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=10
    ).run()
    at.text_input(key="source_folder").set_value(str(folder_b)).run()

    assert not at.exception
    assert any("**7** screenshot(s) indexed" in item.value for item in at.markdown)
    assert not any("**a-" in item.value for item in at.markdown)

    at.text_input(key="screenshot_search").set_value("folder-a").run()
    assert not at.exception
    assert any("**0** screenshot(s) matching your search" in item.value for item in at.markdown)
    assert not any("**a-" in item.value for item in at.markdown)

    at.text_input(key="source_folder").set_value(str(folder_a)).run()
    assert not at.exception
    assert any("**3** screenshot(s) matching your search" in item.value for item in at.markdown)
    assert any("**a-0.png**" in item.value for item in at.markdown)


def test_copy_is_default_and_move_requires_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder = tmp_path / "selected"
    folder.mkdir()
    image_path = folder / "approved.png"
    image_path.write_bytes(b"image bytes")
    stat = image_path.stat()
    with storage.connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO screenshots(
                path, file_size, modified_at, content_hash, width, height,
                ocr_text, ocr_status, indexed_at
            ) VALUES (?, ?, ?, ?, 20, 20, ?, 'complete', '2026-01-01')
            """,
            (
                str(image_path.resolve()),
                stat.st_size,
                stat.st_mtime,
                storage._hash_file(image_path),
                "Python traceback error found in application",
            ),
        )
    item = storage.get_item(cursor.lastrowid)
    prompt = "Organize screenshots into Work and Needs Review."
    model = os.environ.get("SCREENARCH_OLLAMA_MODEL", ollama_client.DEFAULT_MODEL)
    input_hash = classification._input_hash(item, classification.DEFAULT_MAX_OCR_CHARS)
    storage.save_classification(
        item["id"],
        classification._digest(prompt),
        prompt,
        model,
        input_hash,
        {
            "category": "Work",
            "reason": "The OCR supports this category.",
            "evidence": "Python traceback error",
            "confidence": 0.9,
            "needs_review": False,
            "status": classification.STATUS_CLASSIFIED,
        },
    )
    saved = storage.get_saved_classification(
        item["id"], classification._digest(prompt), model, input_hash
    )
    storage.save_review_decision(saved["id"], "Work", True)

    at = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=10
    ).run()
    at.text_input(key="source_folder").set_value(str(folder)).run()
    at.text_area(key="organization_prompt").set_value(prompt).run()

    assert not at.exception
    assert any(
        "For a Needs Review or miscategorized result" in item.value
        for item in at.caption
    )
    operation = next(widget for widget in at.selectbox if widget.label == "File operation")
    assert operation.value == "Copy (keep originals)"
    assert not any(
        widget.label.startswith("I understand that moving removes")
        for widget in at.checkbox
    )

    operation.select("Move (remove originals)").run()
    assert not at.exception
    assert any(
        widget.label.startswith("I understand that moving removes")
        for widget in at.checkbox
    )

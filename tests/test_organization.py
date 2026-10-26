import hashlib

import organization
import storage


def _add_result(path, category="Work", status="classified", needs_review=False, approved=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = path.name.encode("utf-8")
    path.write_bytes(content)
    file_stat = path.stat()
    content_hash = hashlib.sha256(content).hexdigest()
    with storage.connect() as connection:
        cursor = connection.execute("""
            INSERT INTO screenshots(
                path, file_size, modified_at, content_hash, width, height,
                ocr_text, ocr_status, indexed_at
            )
            VALUES (?, ?, ?, ?, 100, 80, 'sample OCR text', 'complete', '2026-01-01')
        """, (
            str(path.resolve()),
            file_stat.st_size,
            file_stat.st_mtime,
            content_hash,
        ))
    screenshot_id = cursor.lastrowid
    prompt_hash = f"prompt-{screenshot_id}"
    input_hash = f"input-{screenshot_id}"
    storage.save_classification(
        screenshot_id,
        prompt_hash,
        "Organize screenshots into Work and Needs Review.",
        "gemma3:4b",
        input_hash,
        {
            "category": category,
            "reason": "The OCR supports this category.",
            "evidence": "sample OCR text",
            "confidence": 0.9,
            "needs_review": needs_review,
            "status": status,
        },
    )
    saved = storage.get_saved_classification(
        screenshot_id, prompt_hash, "gemma3:4b", input_hash
    )
    if approved:
        storage.save_review_decision(saved["id"], category, True)
    return {"id": saved["id"], "path": str(path.resolve())}


def test_approval_and_category_edit_persist_and_bulk_approval_is_conservative(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    suitable = _add_result(tmp_path / "source" / "good.png")
    low_confidence = _add_result(
        tmp_path / "source" / "low.png",
        status="low_confidence",
        needs_review=True,
    )
    review = _add_result(
        tmp_path / "source" / "review.png",
        category="Needs Review",
        needs_review=True,
    )

    approved_count = storage.approve_suitable_classifications(
        [suitable["id"], low_confidence["id"], review["id"]]
    )
    storage.save_review_decision(low_confidence["id"], "Personal", True)
    approved = {row["id"]: row for row in storage.get_approved_classifications()}

    assert approved_count == 1
    assert approved[suitable["id"]]["category"] == "Work"
    assert approved[low_confidence["id"]]["category"] == "Personal"
    assert review["id"] not in approved


def test_copy_preserves_original_and_repeated_run_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    source_root = tmp_path / "source"
    row = _add_result(source_root / "capture.png", approved=True)
    output_root = source_root / "ScreenArch_Organized"

    first = organization.organize_approved([row], output_root, source_root=source_root)
    second = organization.organize_approved([row], output_root, source_root=source_root)

    assert len(first["copied"]) == 1
    destination = tmp_path / "source" / "ScreenArch_Organized" / "Work" / "capture.png"
    assert destination.read_bytes() == row["path"].split("\\")[-1].encode("utf-8")
    assert (source_root / "capture.png").is_file()
    assert len(second["skipped"]) == 1
    assert len(list(output_root.rglob("capture*.png"))) == 1


def test_organization_uses_only_approved_results_from_active_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    folder_a = tmp_path / "folder-a"
    folder_b = tmp_path / "folder-b"
    result_a = _add_result(folder_a / "a.png", approved=True)
    result_b = _add_result(folder_b / "b.png", approved=True)

    result = organization.organize_approved(
        [result_a, result_b],
        tmp_path / "organized",
        source_root=str(folder_a).upper(),
    )

    assert len(result["copied"]) == 1
    assert result["copied"][0]["source"] == result_a["path"]
    assert result["skipped"] == []
    assert (tmp_path / "organized" / "Work" / "a.png").is_file()
    assert not (tmp_path / "organized" / "Work" / "b.png").exists()


def test_duplicate_filenames_receive_distinct_destinations_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    first = _add_result(tmp_path / "source-a" / "capture.png", approved=True)
    second = _add_result(tmp_path / "source-b" / "capture.png", approved=True)
    output_root = tmp_path / "organized"
    existing = output_root / "Work" / "capture.png"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"pre-existing file")

    result = organization.organize_approved([first, second], output_root)
    destinations = [row["destination"] for row in result["copied"]]

    assert len(destinations) == 2
    assert len(set(destinations)) == 2
    assert existing.read_bytes() == b"pre-existing file"
    assert (output_root / "Work" / "capture (2).png").is_file()
    assert (output_root / "Work" / "capture (3).png").is_file()


def test_source_changed_after_classification_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    source = tmp_path / "source" / "capture.png"
    row = _add_result(source, approved=True)
    source.write_bytes(b"changed.png")

    result = organization.organize_approved([row], tmp_path / "organized")
    operation = storage.get_organization_operation(
        row["id"], "copy", str((tmp_path / "organized").resolve())
    )

    assert result["copied"] == []
    assert "changed since its classification" in result["skipped"][0]["reason"]
    assert operation["status"] == "skipped"
    assert not (tmp_path / "organized").exists()


def test_copy_filesystem_error_is_reported_and_retry_can_succeed(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    row = _add_result(tmp_path / "source" / "capture.png", approved=True)
    output_root = tmp_path / "source" / "Sorted"
    original_copy = organization._copy_exclusive

    def fail_copy(*args, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(organization, "_copy_exclusive", fail_copy)
    failed = organization.organize_approved([row], output_root)
    operation = storage.get_organization_operation(row["id"], "copy", str(output_root.resolve()))
    assert "access denied" in failed["failed"][0]["reason"]
    assert operation["status"] == "failed"

    monkeypatch.setattr(organization, "_copy_exclusive", original_copy)
    retried = organization.organize_approved([row], output_root)
    operation = storage.get_organization_operation(row["id"], "copy", str(output_root.resolve()))
    assert len(retried["copied"]) == 1
    assert operation["status"] == "succeeded"


def test_move_requires_approval_and_removes_source_only_after_verified_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    row = _add_result(tmp_path / "source" / "capture.png", approved=True)
    output_root = tmp_path / "organized"

    result = organization.organize_approved(
        [row], output_root, operation="move", source_root=tmp_path / "source"
    )
    repeated = organization.organize_approved(
        [row], output_root, operation="move", source_root=tmp_path / "source"
    )

    assert len(result["moved"]) == 1
    assert not (tmp_path / "source" / "capture.png").exists()
    assert (output_root / "Work" / "capture.png").is_file()
    assert len(repeated["skipped"]) == 1
    assert storage.list_items(folder=tmp_path / "source") == []
    assert storage.list_classification_candidates(folder=tmp_path / "source") == []
    with storage.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM screenshot_classifications"
        ).fetchone()[0] == 1

    scan = storage.index_folder(tmp_path / "source")

    assert scan["new"] == 0
    assert storage.list_items(folder=tmp_path / "source") == []
    assert storage.get_item(row["id"])["is_removed"] == 1


def test_unapproved_review_and_failed_classifications_are_never_organized(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    unapproved = _add_result(tmp_path / "source" / "unapproved.png")
    needs_review = _add_result(
        tmp_path / "source" / "review.png",
        category="Needs Review",
        needs_review=True,
        approved=True,
    )
    model_error = _add_result(
        tmp_path / "source" / "network.png",
        status="model_error",
        approved=True,
    )
    invalid_output = _add_result(
        tmp_path / "source" / "invalid.png",
        status="invalid_model_output",
        approved=True,
    )
    rows = [unapproved, needs_review, model_error, invalid_output]

    result = organization.organize_approved(rows, tmp_path / "organized")

    assert result["copied"] == []
    assert result["moved"] == []
    assert len(result["skipped"]) == 4
    assert all(
        storage.get_organization_operation(
            row["id"], "copy", str((tmp_path / "organized").resolve())
        )["status"] == "skipped"
        for row in rows
    )
    assert not (tmp_path / "organized").exists()

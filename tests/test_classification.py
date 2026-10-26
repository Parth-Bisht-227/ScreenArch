import json
import sqlite3
import time
from urllib.error import URLError

from PIL import Image

import classification
import ollama_client
import storage


PROMPT = "Organize screenshots into Work, Personal, and Needs Review. Do not delete anything."
MODEL = "gemma3:4b"


def _add_screenshot(path, text, ocr_status="complete"):
    with storage.connect() as connection:
        cursor = connection.execute("""
            INSERT INTO screenshots(
                path, file_size, modified_at, content_hash, width, height,
                ocr_text, ocr_status, indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(path), 123, 1.0, "hash", 100, 80, text, ocr_status, "2026-01-01T00:00:00+00:00"))
    return cursor.lastrowid


def _successful_response(
    category="Work",
    evidence="Python traceback error",
    confidence=0.91,
    needs_review=False,
):
    return json.dumps({
        "category": category,
        "reason": "The OCR contains a code error.",
        "evidence": evidence,
        "confidence": confidence,
        "needs_review": needs_review,
    })


def test_categories_are_read_from_the_user_instruction():
    prompt = (
        "Organize my screenshots into Coding & Development, College & Study, Payments & Receipts, "
        "Shopping, Travel, Memes & Entertainment, and Personal. Put uncertain screenshots into "
        "Needs Review. Do not delete anything."
    )

    assert classification.categories_from_prompt(prompt) == [
        "Coding & Development",
        "College & Study",
        "Payments & Receipts",
        "Shopping",
        "Travel",
        "Memes & Entertainment",
        "Personal",
        "Needs Review",
    ]
    assert classification.categories_from_prompt(
        "Sort screenshots into Food and Drink, Home, and Needs Review."
    ) == ["Food and Drink", "Home", "Needs Review"]
    assert classification.categories_from_prompt(
        "Organize screenshots into Mercari Job prep and informational content folders only."
    ) == ["Mercari Job prep", "informational content"]


def test_parse_valid_response_uses_exact_allowed_category():
    result = classification.parse_classification_response(
        _successful_response(),
        ["work", "Personal", "Needs Review"],
        "Python traceback error at line 10",
    )

    assert result == {
        "category": "work",
        "reason": "The OCR contains a code error.",
        "evidence": "Python traceback error",
        "confidence": 0.91,
        "needs_review": False,
        "status": classification.STATUS_CLASSIFIED,
    }


def test_evidence_accepts_case_whitespace_and_punctuation_differences():
    result = classification.parse_classification_response(
        _successful_response(evidence="PYTHON traceback error"),
        ["Work"],
        "Python,\n  traceback — error at line 10",
    )

    assert result["status"] == classification.STATUS_CLASSIFIED
    assert result["evidence"] == "PYTHON traceback error"


def test_malformed_json_and_missing_fields_are_marked_for_review():
    malformed = classification.parse_classification_response(
        "{not json", ["Work"], "some sufficient OCR text"
    )
    missing = classification.parse_classification_response(
        '{"category":"Work"}', ["Work"], "some sufficient OCR text"
    )

    assert malformed["needs_review"] is True
    assert malformed["status"] == classification.STATUS_INVALID_MODEL_OUTPUT
    assert "valid JSON" in malformed["reason"]
    assert missing["needs_review"] is True
    assert "missing required fields" in missing["reason"]


def test_invalid_category_and_evidence_are_marked_for_review():
    out_of_scheme = json.dumps({
        "category": "Made Up",
        "reason": "Reason",
        "evidence": "Python traceback error",
        "confidence": 0.8,
        "needs_review": False,
    })
    invented_evidence = json.dumps({
        "category": "Work",
        "reason": "Reason",
        "evidence": "not in OCR text",
        "confidence": 0.8,
        "needs_review": False,
    })

    assert classification.parse_classification_response(
        out_of_scheme, ["Work"], "Python traceback error"
    )["needs_review"] is True
    assert classification.parse_classification_response(
        invented_evidence, ["Work"], "Python traceback error"
    )["status"] == classification.STATUS_INVALID_EVIDENCE


def test_supported_sample_content_is_classifiable():
    examples = [
        (
            "MERCARI mle projects.jpg",
            "The project uses a MySQL assistant with LangChain and SQLAlchemy.",
            "Technical Projects",
            "MySQL assistant with LangChain and SQLAlchemy",
        ),
        (
            "Gemini_Generated_Image_7ds4tm7ds4tm7ds4.png",
            "Affordable housing and slum transformation through community development.",
            "Housing & Community",
            "Affordable housing and slum transformation",
        ),
    ]
    categories = ["Technical Projects", "Housing & Community", "Needs Review"]

    for filename, ocr_text, category, evidence in examples:
        result = classification.parse_classification_response(
            _successful_response(category, evidence),
            categories,
            ocr_text,
        )
        assert result["category"] == category, filename
        assert result["status"] == classification.STATUS_CLASSIFIED, filename
        assert result["evidence"]


def test_low_confidence_is_explicitly_review_required():
    result = classification.parse_classification_response(
        _successful_response(confidence=0.49),
        ["Work"],
        "Python traceback error",
    )

    assert result["status"] == classification.STATUS_LOW_CONFIDENCE
    assert result["needs_review"] is True
    assert "below the 0.50 review threshold" in result["reason"]


def test_empty_or_insufficient_ocr_is_reviewed_without_model_request(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "empty.png", "")
    _add_screenshot(tmp_path / "short.png", "Too little text")
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])
    monkeypatch.setattr(
        ollama_client,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model should not be called")),
    )

    result = classification.run_classification(PROMPT, MODEL)
    saved = classification.get_results(PROMPT, MODEL)

    assert result["classified"] == 2
    assert result["needs_review"] == 2
    assert all(
        row["category"] == "Needs Review"
        and row["needs_review"]
        and row["result_status"] == classification.STATUS_INSUFFICIENT_OCR
        for row in saved
    )


def test_invalid_evidence_is_retried_once_and_corrected_result_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(
        tmp_path / "mysql.png",
        "MySQL assistant built with LangChain and SQLAlchemy for projects",
    )
    responses = iter([
        _successful_response(evidence="MySQL integration for a database helper"),
        _successful_response(evidence="MySQL assistant built with LangChain"),
    ])
    prompts = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])

    def generate(model, prompt, system):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(ollama_client, "generate", generate)

    result = classification.run_classification(PROMPT, MODEL)
    saved = classification.get_results(PROMPT, MODEL)[0]

    assert len(prompts) == 2
    assert "Correction:" in prompts[1]
    assert saved["result_status"] == classification.STATUS_CLASSIFIED
    assert saved["evidence"] == "MySQL assistant built with LangChain"
    assert result["classified"] == 1
    assert result["failed"] == 0
    assert len(saved["model_request_durations"]) == 2


def test_invalid_evidence_retry_is_bounded_and_preserves_review_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "invented.png", "MySQL assistant uses LangChain.")
    calls = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])
    monkeypatch.setattr(
        ollama_client,
        "generate",
        lambda *args, **kwargs: calls.append(args) or _successful_response(
            evidence="invented database platform"
        ),
    )

    classification.run_classification(PROMPT, MODEL)
    saved = classification.get_results(PROMPT, MODEL)[0]

    assert len(calls) == 2
    assert saved["category"] == "Needs Review"
    assert saved["evidence"] == ""
    assert saved["confidence"] == 0
    assert saved["result_status"] == classification.STATUS_INVALID_EVIDENCE
    assert "after one correction attempt" in saved["reason"]


def test_invalid_model_output_gets_one_schema_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "schema.png", "Python traceback error found in application")
    responses = iter(['{"category":"Work"}', _successful_response()])
    prompts = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])

    def generate(model, prompt, system):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(ollama_client, "generate", generate)
    classification.run_classification(PROMPT, MODEL)
    saved = classification.get_results(PROMPT, MODEL)[0]

    assert len(prompts) == 2
    assert "previous response did not follow the required JSON schema" in prompts[1]
    assert saved["result_status"] == classification.STATUS_CLASSIFIED


def test_invalid_model_output_remains_review_after_bounded_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "schema.png", "Python traceback error found in application")
    calls = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])
    monkeypatch.setattr(
        ollama_client,
        "generate",
        lambda *args, **kwargs: calls.append(args) or '{"category":"Work"}',
    )
    classification.run_classification(PROMPT, MODEL)
    saved = classification.get_results(PROMPT, MODEL)[0]

    assert len(calls) == 2
    assert saved["result_status"] == classification.STATUS_INVALID_MODEL_OUTPUT
    assert "after one correction attempt" in saved["reason"]


def test_classification_progress_and_request_timing_are_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "timed.png", "Python traceback error found in application")
    events = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])

    def timed_generate(*args, **kwargs):
        time.sleep(0.01)
        return _successful_response()

    monkeypatch.setattr(ollama_client, "generate", timed_generate)
    result = classification.run_classification(
        PROMPT,
        MODEL,
        progress_callback=events.append,
    )
    saved = classification.get_results(PROMPT, MODEL)[0]

    assert [event["event"] for event in events] == ["started", "completed"]
    assert events[-1]["completed"] == events[-1]["total"] == 1
    assert events[-1]["request_seconds"] >= 0.01
    assert len(events[-1]["request_durations_seconds"]) == 1
    assert result["elapsed_seconds"] >= events[-1]["request_seconds"]
    assert saved["model_duration_seconds"] >= 0.01
    assert len(saved["model_request_durations"]) == 1


def test_ollama_unreachable_endpoint_is_actionable(monkeypatch):
    monkeypatch.setattr(ollama_client, "get_base_url", lambda: ollama_client.DEFAULT_BASE_URL)
    monkeypatch.setattr(
        ollama_client,
        "build_opener",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("connection refused")),
    )

    try:
        ollama_client.list_models()
    except ollama_client.OllamaError as exc:
        assert "Start Ollama" in str(exc)
    else:
        raise AssertionError("Expected OllamaError")


def test_missing_model_is_reported_without_download(monkeypatch):
    monkeypatch.setattr(ollama_client, "list_models", lambda timeout=10: ["gemma3:1b"])

    try:
        ollama_client.require_model_available(MODEL)
    except ollama_client.OllamaError as exc:
        assert "not installed" in str(exc)
        assert "never downloads" in str(exc)
    else:
        raise AssertionError("Expected OllamaError")


def test_ollama_non_object_response_is_rejected(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"[]"

    class Opener:
        def open(self, request, timeout):
            return Response()

    monkeypatch.setattr(ollama_client, "get_base_url", lambda: ollama_client.DEFAULT_BASE_URL)
    monkeypatch.setattr(ollama_client, "build_opener", lambda *args: Opener())

    try:
        ollama_client.list_models()
    except ollama_client.OllamaError as exc:
        assert "expected an object response" in str(exc)
    else:
        raise AssertionError("Expected OllamaError")


def test_remote_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("SCREENARCH_OLLAMA_URL", "http://example.invalid:11434")

    try:
        ollama_client.get_base_url()
    except ollama_client.OllamaError as exc:
        assert "only sends OCR to a local Ollama endpoint" in str(exc)
    else:
        raise AssertionError("Expected OllamaError")


def test_non_gemma_model_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ollama_client,
        "list_models",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must reject before lookup")),
    )

    try:
        ollama_client.require_model_available("llama3:8b")
    except ollama_client.OllamaError as exc:
        assert "requires a locally installed Gemma model" in str(exc)
    else:
        raise AssertionError("Expected OllamaError")


def test_classification_persists_and_skips_unchanged_results(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "code.png", "Python traceback error found in application")
    generate_calls = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])
    monkeypatch.setattr(
        ollama_client,
        "generate",
        lambda *args, **kwargs: generate_calls.append(args) or _successful_response(),
    )

    first = classification.run_classification(PROMPT, MODEL)
    second = classification.run_classification(PROMPT, MODEL)
    results = classification.get_results(PROMPT, MODEL)

    assert first["classified"] == 1
    assert second["skipped"] == 1
    assert len(generate_calls) == 1
    assert len(results) == 1
    assert results[0]["category"] == "Work"
    assert results[0]["prompt"] == PROMPT
    with storage.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM screenshot_classifications").fetchone()[0] == 1


def test_cached_results_preserve_review_approval_until_forced_reclassification(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "code.png", "Python traceback error found in application")
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])
    monkeypatch.setattr(ollama_client, "generate", lambda *args, **kwargs: _successful_response())

    classification.run_classification(PROMPT, MODEL)
    initial = classification.get_results(PROMPT, MODEL)[0]
    storage.save_review_decision(initial["id"], "Personal", True)
    cached = classification.run_classification(PROMPT, MODEL)
    cached_row = classification.get_results(PROMPT, MODEL)[0]
    classification.run_classification(PROMPT, MODEL, force=True)
    refreshed_row = classification.get_results(PROMPT, MODEL)[0]

    assert cached["skipped"] == 1
    assert cached_row["category"] == "Personal"
    assert cached_row["approved"] is True
    assert refreshed_row["category"] == "Work"
    assert refreshed_row["approved"] is False


def test_changed_prompt_gets_separate_results_and_forced_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "code.png", "Python traceback error found in application")
    calls = []
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])
    def generate(model, model_prompt, system):
        calls.append((model, model_prompt, system))
        category = "Coding" if "Sort screenshots" in model_prompt else "Work"
        return _successful_response(category=category)

    monkeypatch.setattr(ollama_client, "generate", generate)
    changed_prompt = "Sort screenshots into Coding, Home, and Needs Review."

    classification.run_classification(PROMPT, MODEL)
    changed_result = classification.run_classification(changed_prompt, MODEL)
    forced_result = classification.run_classification(PROMPT, MODEL, force=True)

    assert changed_result["classified"] == 1
    assert forced_result["classified"] == 1
    assert len(calls) == 3
    assert classification.get_results(PROMPT, MODEL)[0]["prompt"] == PROMPT
    assert classification.get_results(changed_prompt, MODEL)[0]["prompt"] == changed_prompt
    with storage.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM screenshot_classifications").fetchone()[0] == 2


def test_partial_model_failure_keeps_successful_results(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    _add_screenshot(tmp_path / "success.png", "Python traceback error found in application")
    _add_screenshot(tmp_path / "failure.png", "Python traceback error found in application")
    calls = iter([_successful_response(), ollama_client.OllamaError("temporary model failure")])
    monkeypatch.setattr(ollama_client, "require_model_available", lambda model: [MODEL])

    def generate(*args, **kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(ollama_client, "generate", generate)

    result = classification.run_classification(PROMPT, MODEL)

    assert result["classified"] == 1
    assert len(result["errors"]) == 1
    assert result["errors"][0]["path"].endswith("failure.png")
    saved = classification.get_results(PROMPT, MODEL)
    assert len(saved) == 2
    failure = next(row for row in saved if row["path"].endswith("failure.png"))
    assert failure["result_status"] == classification.STATUS_MODEL_ERROR
    assert "temporary model failure" in failure["reason"]


def test_existing_database_migration_preserves_ocr_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "screenshots.db")
    storage.init_db()
    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (12, 10), "white").save(image_path)
    with storage.connect() as connection:
        connection.execute("""
            INSERT INTO screenshots(path, file_size, modified_at, content_hash, width, height,
                                    ocr_text, ocr_status, indexed_at)
            VALUES (?, 1, 1, 'hash', 12, 10, 'existing receipt text', 'complete', '2026-01-01')
        """, (str(image_path),))

    storage.init_db()

    assert storage.list_items()[0]["ocr_text"] == "existing receipt text"
    with storage.connect() as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "screenshot_classifications" in tables


def test_legacy_classification_rows_migrate_without_losing_approval_safe_defaults(tmp_path, monkeypatch):
    database = tmp_path / "legacy.db"
    monkeypatch.setattr(storage, "DB_PATH", database)
    with sqlite3.connect(database) as connection:
        connection.execute("""
            CREATE TABLE screenshots (
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
        connection.execute("""
            CREATE TABLE screenshot_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screenshot_id INTEGER NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                category TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence TEXT NOT NULL,
                confidence REAL NOT NULL,
                needs_review INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(screenshot_id, prompt_hash, model, input_hash)
            )
        """)
        connection.execute("""
            INSERT INTO screenshots(path, file_size, modified_at, ocr_text, ocr_status, indexed_at)
            VALUES ('legacy.png', 1, 1.0, 'legacy OCR content', 'complete', '2026-01-01')
        """)
        connection.execute("""
            INSERT INTO screenshot_classifications(
                screenshot_id, prompt_hash, prompt, model, input_hash, category,
                reason, evidence, confidence, needs_review, created_at
            )
            VALUES (1, 'prompt', 'Organize into Work.', 'gemma3:4b', 'input',
                    'Work', 'Legacy reason', 'legacy OCR', 0.8, 0, '2026-01-01')
        """)

    storage.init_db()
    with storage.connect() as connection:
        migrated = connection.execute(
            "SELECT * FROM screenshot_classifications WHERE id = 1"
        ).fetchone()

    assert migrated["category"] == "Work"
    assert migrated["evidence"] == "legacy OCR"
    assert migrated["result_status"] == classification.STATUS_CLASSIFIED
    assert migrated["model_duration_seconds"] == 0
    assert migrated["model_request_durations"] == "[]"
    assert migrated["approved"] == 0
    assert migrated["approved_category"] is None

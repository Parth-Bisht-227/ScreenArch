"""Explicit, collision-safe file organization for approved classifications."""

import hashlib
import os
import re
import shutil
from pathlib import Path

import storage

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_FAILED_CLASSIFICATION_STATUSES = {"model_error", "invalid_model_output"}


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_category_name(category):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", category).strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        cleaned = "Uncategorized"
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def _within(path, root):
    path_identity = storage.normalize_path(path)
    root_identity = storage.normalize_path(root)
    return (
        path_identity == root_identity
        or path_identity.startswith(root_identity.rstrip("/") + "/")
    )


def _next_destination(directory, filename):
    original = directory / filename
    if not original.exists():
        return original
    stem, suffix = original.stem, original.suffix
    index = 2
    while True:
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _record_skip(result, classification_id, source, output_root, operation, reason):
    storage.save_organization_operation(
        classification_id,
        str(source),
        str(output_root),
        "",
        operation,
        "",
        "skipped",
        reason,
    )
    result["skipped"].append({"source": str(source), "reason": reason})


def _copy_exclusive(source_path, destination):
    directory = destination.parent
    attempted = destination
    while True:
        try:
            descriptor = os.open(
                attempted,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
        except FileExistsError:
            attempted = _next_destination(directory, destination.name)
            continue

        try:
            with source_path.open("rb") as source, os.fdopen(descriptor, "wb") as target:
                shutil.copyfileobj(source, target)
            shutil.copystat(source_path, attempted)
            if _file_hash(source_path) != _file_hash(attempted):
                raise OSError("Source changed while it was being copied.")
        except Exception:
            try:
                attempted.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return attempted


def organize_approved(results, output_dir, operation="copy", source_root=None):
    if operation not in {"copy", "move"}:
        raise ValueError("Organization operation must be 'copy' or 'move'.")

    output_root = Path(output_dir).expanduser().resolve()
    source_root_path = Path(source_root).expanduser().resolve() if source_root else None
    if source_root_path and output_root == source_root_path:
        raise ValueError("Choose an output directory inside or outside the source folder, not the source folder itself.")
    result = {"copied": [], "moved": [], "skipped": [], "failed": []}
    approved_by_id = {
        row["id"]: row
        for row in storage.get_approved_classifications(folder=source_root_path)
    }

    for row in results:
        source = Path(row["path"]).expanduser().resolve()
        classification_id = row["id"]
        approved_row = approved_by_id.get(classification_id)
        if source_root_path and not _within(source, source_root_path):
            continue
        if row.get("result_status") in _FAILED_CLASSIFICATION_STATUSES:
            _record_skip(
                result,
                classification_id,
                source,
                output_root,
                operation,
                f"Classification failed ({row['result_status']}); it cannot be organized.",
            )
            continue
        if row.get("category", "").strip().casefold() == "needs review":
            _record_skip(
                result,
                classification_id,
                source,
                output_root,
                operation,
                "Needs Review items cannot be organized.",
            )
            continue
        if approved_row is None:
            _record_skip(
                result,
                classification_id,
                source,
                output_root,
                operation,
                "Not explicitly approved.",
            )
            continue
        if approved_row["result_status"] in _FAILED_CLASSIFICATION_STATUSES:
            _record_skip(
                result,
                classification_id,
                source,
                output_root,
                operation,
                f"Classification failed ({approved_row['result_status']}); it cannot be organized.",
            )
            continue
        category = approved_row["category"].strip()
        if not category or category.casefold() == "needs review":
            _record_skip(
                result,
                classification_id,
                source,
                output_root,
                operation,
                "Needs Review items cannot be organized.",
            )
            continue
        if _within(source, output_root):
            _record_skip(
                result,
                classification_id,
                source,
                output_root,
                operation,
                "The source is already inside the selected output directory.",
            )
            continue

        destination = ""
        try:
            previous = storage.get_organization_operation(
                classification_id, operation, str(output_root)
            )
            previous_destination = (
                Path(previous["destination_path"])
                if previous and previous["destination_path"]
                else None
            )
            if (
                previous
                and previous["status"] == "succeeded"
                and previous_destination
                and previous_destination.is_file()
                and _file_hash(previous_destination) == previous["file_hash"]
            ):
                if operation == "move" and not source.exists():
                    storage.mark_item_removed(source)
                result["skipped"].append({
                    "source": str(source),
                    "destination": previous["destination_path"],
                    "reason": f"Already {operation}d successfully in a previous run.",
                })
                continue
            if (
                previous
                and previous_destination
                and previous_destination.is_file()
                and _file_hash(previous_destination) == previous["file_hash"]
            ):
                source_hash = ""
                recovered_move = False
                if source.is_file():
                    source_hash = _file_hash(source)
                    if source_hash == previous["file_hash"] and operation == "move":
                        source.unlink()
                        recovered_move = True
                    elif source_hash != previous["file_hash"]:
                        source_hash = ""
                    if operation == "copy" and source_hash != previous["file_hash"]:
                        source_hash = ""
                if not source.exists() or source_hash == previous["file_hash"]:
                    if operation == "move":
                        storage.mark_item_removed(source)
                    storage.save_organization_operation(
                        classification_id,
                        str(source),
                        str(output_root),
                        str(previous_destination),
                        operation,
                        previous["file_hash"],
                        "succeeded",
                    )
                    if recovered_move:
                        result["moved"].append({
                            "source": str(source),
                            "destination": str(previous_destination),
                        })
                    elif operation == "move" and not source.exists():
                        result["skipped"].append({
                            "source": str(source),
                            "destination": str(previous_destination),
                            "reason": "Recovered a previously completed move.",
                        })
                    else:
                        result["skipped"].append({
                            "source": str(source),
                            "destination": str(previous_destination),
                            "reason": f"Recovered a previously completed {operation} operation.",
                        })
                    continue
            if not source.is_file():
                raise FileNotFoundError(f"Source file not found: {source}")
            source_stat = source.stat()
            file_hash = _file_hash(source)
            if (
                source_stat.st_size != approved_row["source_file_size"]
                or source_stat.st_mtime != approved_row["source_modified_at"]
                or (
                    approved_row["source_content_hash"]
                    and file_hash != approved_row["source_content_hash"]
                )
            ):
                reason = (
                    "Source changed since its classification was indexed; rescan and reclassify before organizing."
                )
                storage.save_organization_operation(
                    classification_id,
                    str(source),
                    str(output_root),
                    "",
                    operation,
                    file_hash,
                    "skipped",
                    reason,
                )
                result["skipped"].append({
                    "source": str(source),
                    "reason": reason,
                })
                continue
            category_dir = output_root / _safe_category_name(category)
            category_dir.mkdir(parents=True, exist_ok=True)
            if category_dir.is_symlink():
                raise OSError(f"Category folder must not be a symbolic link: {category_dir}")

            destination = (
                Path(previous["destination_path"])
                if previous and previous["file_hash"] == file_hash
                else _next_destination(category_dir, source.name)
            )

            storage.save_organization_operation(
                classification_id,
                str(source),
                str(output_root),
                str(destination),
                operation,
                file_hash,
                "in_progress",
            )
            destination = _copy_exclusive(source, destination)
            if operation == "move":
                source.unlink()
                storage.mark_item_removed(source)
            storage.save_organization_operation(
                classification_id,
                str(source),
                str(output_root),
                str(destination),
                operation,
                file_hash,
                "succeeded",
            )
            result["copied" if operation == "copy" else "moved"].append({
                "source": str(source),
                "destination": str(destination),
            })
        except OSError as exc:
            try:
                file_hash = _file_hash(source) if source.is_file() else ""
                storage.save_organization_operation(
                    classification_id,
                    str(source),
                    str(output_root),
                    str(destination),
                    operation,
                    file_hash,
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                )
            except OSError:
                raise
            result["failed"].append({
                "source": str(source),
                "destination": str(destination),
                "reason": f"{type(exc).__name__}: {exc}",
            })

    return result

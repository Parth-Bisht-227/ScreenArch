from pathlib import Path
import hashlib
import os
import queue
import threading
import time

import pandas as pd
import streamlit as st
from PIL import Image
import classification
import ollama_client
import organization
import storage
from storage import init_db, list_items, index_folder, retry_failed_ocr, search_items

st.set_page_config(page_title="ScreenArch", page_icon="🔎", layout="wide")
init_db()


def _concise(text, limit=160):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _start_classification(prompt, model_name, max_ocr_chars, force, folder):
    events = queue.Queue()
    state = {
        "events": events,
        "started_at": time.perf_counter(),
        "completed": 0,
        "total": 0,
        "folder": storage.normalize_path(folder),
        "current_path": "",
        "history": [],
        "finished": False,
        "refreshed": False,
    }
    st.session_state["classification_progress"] = state

    def run():
        try:
            result = classification.run_classification(
                prompt,
                model_name,
                max_ocr_chars=max_ocr_chars,
                force=force,
                progress_callback=events.put,
                folder=folder,
            )
        except Exception as exc:
            events.put({
                "event": "error",
                "message": f"{type(exc).__name__}: {exc}",
            })
        else:
            events.put({"event": "finished", "result": result})

    worker = threading.Thread(target=run, name="screenarch-classification", daemon=True)
    worker.start()


@st.fragment(run_every="1s")
def _classification_progress(render=True):
    state = st.session_state.get("classification_progress")
    if not state:
        return

    while True:
        try:
            event = state["events"].get_nowait()
        except queue.Empty:
            break
        if event["event"] == "started":
            state["current_path"] = event["path"]
            state["total"] = event["total"]
        elif event["event"] == "completed":
            state["completed"] = event["completed"]
            state["total"] = event["total"]
            state["current_path"] = event["path"]
            state["history"].append(event)
        elif event["event"] == "finished":
            state["finished"] = True
            state["result"] = event["result"]
            state["completed"] = event["result"]["completed"]
            state["total"] = event["result"]["total"]
        elif event["event"] == "error":
            state["finished"] = True
            state["error"] = event["message"]

    if not render:
        return

    total = state["total"]
    completed = state["completed"]
    fraction = completed / total if total else 0.0
    elapsed = (
        state["result"]["elapsed_seconds"]
        if state.get("result")
        else time.perf_counter() - state["started_at"]
    )
    if state["finished"]:
        st.progress(min(fraction, 1.0), text=f"Completed {completed} of {total}")
    else:
        st.progress(fraction, text=f"Completed {completed} of {total}")
    if state["finished"]:
        st.caption(f"Batch elapsed time: {elapsed:.1f} seconds")
    else:
        st.caption(f"Elapsed time: {elapsed:.1f} seconds")

    if state["current_path"]:
        label = "Last processed" if state["finished"] else "Currently processing"
        st.write(f"{label}: `{Path(state['current_path']).name}`")
    if state["history"]:
        st.markdown("**Per-image status**")
        for event in state["history"][-10:]:
            status = event["status"].replace("_", " ")
            disposition = "cached" if event.get("skipped") else status
            request_time = event.get("request_seconds", 0.0)
            request_durations = ", ".join(
                f"{duration:.1f}s"
                for duration in event.get("request_durations_seconds", [])
            ) or "no model request"
            st.caption(
                f"{Path(event['path']).name}: {disposition}; "
                f"request times {request_durations} (total {request_time:.1f}s)"
            )
    if state.get("error"):
        st.error(f"Classification did not finish: {state['error']}")
    if state.get("result"):
        result = state["result"]
        st.success(
            f"Batch finished: {result['classified']} saved, {result['skipped']} unchanged, "
            f"{result['needs_review']} need review, {result['failed']} failed."
        )
        if result["errors"]:
            with st.expander("Show classification errors"):
                st.write(result["errors"])
        if not state["refreshed"]:
            state["refreshed"] = True
            st.rerun()
    elif state.get("error") and not state["refreshed"]:
        state["refreshed"] = True
        st.rerun()


with st.sidebar:
    st.header("Index screenshots")
    folder = st.text_input(
        "Folder path",
        placeholder=r"C:\Users\you\Pictures\Screenshots",
        key="source_folder",
    )
    selected_folder = Path(folder).expanduser().resolve() if folder.strip() else None
    active_folder = (
        str(selected_folder)
        if selected_folder is not None and selected_folder.is_dir()
        else None
    )
    st.caption("Start with a small test folder. OCR can take a while on large collections.")
    if st.button("Scan and index folder", type="primary", width="stretch"):
        if not folder.strip():
            st.error("Enter a folder path first.")
        elif active_folder is None:
            st.error("That folder does not exist on this computer.")
        else:
            with st.spinner("Scanning images and extracting text locally..."):
                result = index_folder(selected_folder)
            st.success(
                f"Finished. {result['new']} new, {result['updated']} changed, "
                f"{result['skipped']} unchanged, {result['removed']} removed."
            )
            if result["tesseract_missing"]:
                st.error(
                    "Tesseract OCR is not installed or its executable is not on PATH. "
                    "Install Tesseract, ensure the executable is on PATH (or configure "
                    "pytesseract.pytesseract.tesseract_cmd), then use Retry failed OCR."
                )
            if result["errors"]:
                st.warning(f"{len(result['errors'])} file(s) could not be processed.")
                with st.expander("Show errors"):
                    st.write(result["errors"][:20])

    if st.button("Retry failed OCR", width="stretch", disabled=not active_folder):
        with st.spinner("Retrying failed OCR for this folder..."):
            result = retry_failed_ocr(selected_folder)
        st.success(f"Finished. {result['updated']} retried, {result['skipped']} skipped.")
        if result["tesseract_missing"]:
            st.error(
                "Tesseract OCR is not installed or its executable is not on PATH. "
                "Install Tesseract, ensure the executable is on PATH (or configure "
                "pytesseract.pytesseract.tesseract_cmd), then retry."
            )
        if result["errors"]:
            st.warning(f"{len(result['errors'])} file(s) could not be processed.")
            with st.expander("Show errors"):
                st.write(result["errors"][:20])

    st.divider()
    st.header("Search")
    query = st.text_input(
        "Search screenshots",
        placeholder="e.g. Python error, payment receipt",
        key="screenshot_search",
    )
    st.caption("Keyword search matches text found by OCR.")

st.title("🔎 ScreenArch")
st.subheader("The archaeologist for your screenshots.")
st.caption(
    "A local-first searchable archive for your screenshots. Images stay where they are; "
    "extracted text and metadata are stored in a local SQLite database."
)

st.header("AI organization")
st.caption(
    "Classify with OCR text and file metadata using a local Ollama model. Images and OCR stay on this computer; "
    "the model cannot see the screenshots. Results are recommendations only."
)
organization_prompt = st.text_area(
    "Organization instruction",
    value="",
    placeholder="Sort screenshots into Job Search, Engineering, Content Creation, and Miscellaneous.",
    height=110,
    key="organization_prompt",
)
model_name = st.text_input(
    "Installed local Gemma model name",
    value=os.environ.get("SCREENARCH_OLLAMA_MODEL", ollama_client.DEFAULT_MODEL),
    help="Choose an exact model name shown by `ollama list`. ScreenArch never downloads models.",
)
max_ocr_chars = st.slider(
    "Maximum OCR characters sent per screenshot",
    min_value=1000,
    max_value=20000,
    value=classification.DEFAULT_MAX_OCR_CHARS,
    step=1000,
)
candidates = (
    classification.storage.list_classification_candidates(folder=active_folder)
    if active_folder
    else []
)
eligible = classification.eligible_count(candidates)
st.metric("Eligible for OCR-based classification", f"{eligible} of {len(candidates)} indexed screenshots")
st.caption(
    "Screenshots without at least 20 OCR letters/numbers and three words are not sent to the model "
    "and are saved as Needs Review. Confidence is the model's uncalibrated self-assessment."
)

check_col, run_col, force_col = st.columns([1, 1, 2], vertical_alignment="bottom")
progress_state = st.session_state.get("classification_progress")
classification_running = bool(progress_state and not progress_state["finished"])
folder_identity = storage.normalize_path(active_folder) if active_folder else None
active_progress_state = (
    progress_state
    if folder_identity and progress_state and progress_state["folder"] == folder_identity
    else None
)
with check_col:
    if st.button("Check local model", width="stretch"):
        try:
            ollama_client.require_model_available(model_name)
        except ollama_client.OllamaError as exc:
            st.error(str(exc))
        else:
            st.success(f"Local Ollama model '{model_name}' is available.")
with force_col:
    force_classification = st.checkbox("Reclassify unchanged items", value=False)
with run_col:
    if st.button(
        "Run classification",
        type="primary",
        width="stretch",
        disabled=classification_running or not active_folder,
    ):
        _start_classification(
            organization_prompt,
            model_name,
            max_ocr_chars,
            force_classification,
            active_folder,
        )

if active_progress_state and active_progress_state["finished"]:
    if active_progress_state.get("result"):
        result = active_progress_state["result"]
        st.progress(
            min(result["completed"] / result["total"], 1.0) if result["total"] else 0.0,
            text=f"Completed {result['completed']} of {result['total']}",
        )
        st.caption(f"Batch elapsed time: {result['elapsed_seconds']:.1f} seconds")
        st.success(
            f"Batch finished: {result['classified']} saved, {result['skipped']} unchanged, "
            f"{result['needs_review']} need review, {result['failed']} failed."
        )
        for event in active_progress_state["history"][-10:]:
            request_durations = ", ".join(
                f"{duration:.1f}s"
                for duration in event.get("request_durations_seconds", [])
            ) or "no model request"
            st.caption(
                f"{Path(event['path']).name}: {event['status'].replace('_', ' ')}; "
                f"request times {request_durations} "
                f"(total {event.get('request_seconds', 0.0):.1f}s)"
            )
    else:
        st.error(f"Classification did not finish: {active_progress_state['error']}")
elif active_progress_state:
    _classification_progress()
elif progress_state:
    _classification_progress(render=False)
if classification_running and not active_progress_state:
    st.info("Classification is still running for the previously selected folder.")
current_results = (
    classification.get_results(
        organization_prompt, model_name, max_ocr_chars, folder=active_folder
    )
    if active_folder
    else []
)
visible_results_by_path = {}
if current_results:
    st.subheader("Classification results")
    categories = ["All categories"] + sorted({row["category"] for row in current_results}, key=str.casefold)
    category_filter = st.selectbox("Filter by category", categories)
    review_filter = st.selectbox(
        "Filter by review status",
        ["All", "Needs review", "No review needed"],
    )
    visible_results = current_results
    if category_filter != "All categories":
        visible_results = [row for row in visible_results if row["category"] == category_filter]
    if review_filter != "All":
        needs_review = review_filter == "Needs review"
        visible_results = [row for row in visible_results if row["needs_review"] is needs_review]
    st.dataframe(
        [
            {
                "Screenshot": Path(row["path"]).name,
                "Category": row["category"],
                "Status": row["result_status"].replace("_", " "),
                "Review status": "Needs review" if row["needs_review"] else "No review needed",
                "Reason": _concise(row["reason"], 120),
                "OCR evidence": _concise(row["evidence"], 120),
            }
            for row in visible_results
        ],
        alt="Local AI screenshot classification results",
        width="stretch",
        hide_index=True,
    )
    visible_results_by_path = {
        storage.normalize_path(row["path"]): row for row in visible_results
    }

    st.subheader("Review and approve")
    st.caption(
        "For a Needs Review or miscategorized result, choose its correct Category, check Approved only after "
        "you verify it, and save the form. The approved category becomes its destination folder; unapproved "
        f"items stay untouched. Low confidence below {classification.LOW_CONFIDENCE_THRESHOLD:.1f} requires review. "
        "'Approve all suitable results' excludes review-required items and Needs Review categories."
    )
    try:
        prompt_categories = classification.categories_from_prompt(organization_prompt)
    except classification.PromptCategoryError:
        prompt_categories = []
    category_options = list(dict.fromkeys(
        prompt_categories + [row["category"] for row in current_results] + [classification.REVIEW_CATEGORY]
    ))
    editor_rows = [
        {
            "ID": row["id"],
            "Screenshot": Path(row["path"]).name,
            "Category": row["category"],
            "Approved": row["approved"],
            "Classification status": row["result_status"].replace("_", " "),
            "Reason": row["reason"],
        }
        for row in current_results
    ]
    editor_key = "review_editor_" + hashlib.sha256(
        f"{organization_prompt}\n{model_name}\n{','.join(str(row['id']) for row in current_results)}".encode("utf-8")
    ).hexdigest()[:12]
    with st.form("review_decisions"):
        edited = st.data_editor(
            pd.DataFrame(editor_rows),
            key=editor_key,
            num_rows="fixed",
            hide_index=True,
            width="stretch",
            column_config={
                "ID": None,
                "Screenshot": st.column_config.TextColumn(disabled=True),
                "Category": st.column_config.SelectboxColumn(
                    options=category_options,
                    required=True,
                ),
                "Approved": st.column_config.CheckboxColumn(),
                "Classification status": st.column_config.TextColumn(disabled=True),
                "Reason": st.column_config.TextColumn(disabled=True),
            },
            disabled=["ID", "Screenshot", "Classification status", "Reason"],
        )
        save_decisions = st.form_submit_button("Save category and approval changes")
    if save_decisions:
        for row in edited.to_dict("records"):
            selected_category = str(row["Category"])
            approved = bool(row["Approved"]) and (
                selected_category.casefold() != classification.REVIEW_CATEGORY.casefold()
            )
            storage.save_review_decision(
                row["ID"], selected_category, approved, folder=active_folder
            )
        st.success("Review decisions saved.")
        st.rerun()

    if st.button("Approve all suitable results", width="stretch"):
        suitable_ids = [
            row["id"]
            for row in current_results
            if row["result_status"] == classification.STATUS_CLASSIFIED
            and not row["needs_review"]
            and row["category"].casefold() != classification.REVIEW_CATEGORY.casefold()
        ]
        approved_count = storage.approve_suitable_classifications(
            suitable_ids, folder=active_folder
        )
        st.success(f"Approved {approved_count} suitable result(s). Review-required items were not approved.")
        st.session_state.pop(editor_key, None)
        st.rerun()

    st.subheader("Organize approved screenshots")
    st.caption(
        "The default is to copy approved screenshots into category folders under "
        "ScreenArch_Organized, keeping the originals. Moving is available with separate "
        "confirmation and is verified before the original path is removed from the active gallery."
    )
    if active_folder:
        source_folder_exists = True
        default_output = str(selected_folder / "ScreenArch_Organized")
        output_key = "organization_output_" + hashlib.sha256(
            storage.normalize_path(selected_folder).encode("utf-8")
        ).hexdigest()[:10]
        output_dir = st.text_input(
            "Output directory",
            value=default_output,
            key=output_key,
        )
    else:
        source_folder_exists = False
        st.warning("Select the source folder in the sidebar before organizing.")
        output_dir = ""
    if folder.strip() and not source_folder_exists:
        st.error("The selected source folder does not exist or is not a directory.")
    operation = st.selectbox(
        "File operation",
        ["Copy (keep originals)", "Move (remove originals)"],
    )
    move_confirmed = False
    if operation.startswith("Move"):
        move_confirmed = st.checkbox(
            "I understand that moving removes the original screenshots after the copy is verified."
        )
    organize_disabled = (
        not active_folder
        or not source_folder_exists
        or not output_dir.strip()
        or (operation.startswith("Move") and not move_confirmed)
    )
    if st.button(
        "Organize approved screenshots",
        type="primary",
        width="stretch",
        disabled=organize_disabled,
    ):
        operation_mode = "move" if operation.startswith("Move") else "copy"
        try:
            organization_result = organization.organize_approved(
                current_results,
                output_dir,
                operation=operation_mode,
                source_root=active_folder,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.write(
                f"Copied {len(organization_result['copied'])}; moved {len(organization_result['moved'])}; "
                f"skipped {len(organization_result['skipped'])}; failed {len(organization_result['failed'])}."
            )
            for label, entries in (
                ("Copied", organization_result["copied"]),
                ("Moved", organization_result["moved"]),
                ("Skipped", organization_result["skipped"]),
                ("Failed", organization_result["failed"]),
            ):
                if entries:
                    with st.expander(f"{label} files ({len(entries)})", expanded=True):
                        st.write(entries)
elif organization_prompt.strip():
    st.info(
        "No results yet for this exact prompt and model. Previous results are retained in SQLite, "
        "but are not presented as if they match changed settings."
    )

items = (
    search_items(query, folder=active_folder)
    if query.strip() and active_folder
    else list_items(folder=active_folder)
    if active_folder
    else []
)
st.write(f"**{len(items)}** screenshot(s) {'matching your search' if query.strip() else 'indexed'}")

if not items:
    if not active_folder:
        st.info("Select an existing source folder to view or search its indexed screenshots.")
    else:
        st.info("No screenshots to show yet. Click **Scan and index folder** to index this folder.")
else:
    cols_per_row = 3
    for start in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[start:start + cols_per_row]):
            with col:
                path = Path(item["path"])
                result = visible_results_by_path.get(storage.normalize_path(path))
                if path.exists():
                    try:
                        img = Image.open(path)
                        img.thumbnail((600, 400))
                        st.image(img, width="stretch")
                    except Exception:
                        st.warning("Could not display this image.")
                else:
                    st.warning("Original image not found at its indexed path.")
                st.markdown(f"**{path.name}**")
                st.caption(f"{item['width']} × {item['height']} · OCR: {item['ocr_status']}")
                if result:
                    review_status = "Needs review" if result["needs_review"] else "No review needed"
                    st.markdown(f"**{result['category']}** · {review_status}")
                    st.caption(_concise(result["reason"]))
                    if result["evidence"]:
                        st.caption(f"Evidence: {_concise(result['evidence'])}")
                if item["ocr_text"]:
                    with st.expander("View extracted text"):
                        st.text(item["ocr_text"])
                else:
                    st.caption("No text extracted.")
                with st.expander("Details"):
                    st.code(item["path"])
                    st.write(f"Indexed: {item['indexed_at']}")
                    if st.button("Open original file location", key=f"open_{item['id']}"):
                        st.info(
                            "Copy the path above into File Explorer. Copy is the default; "
                            "moving requires a separate selection and confirmation."
                        )

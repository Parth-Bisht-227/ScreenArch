from pathlib import Path
import streamlit as st
from PIL import Image
from storage import init_db, list_items, index_folder, retry_failed_ocr, search_items, get_item

st.set_page_config(page_title="Screenshot Archaeologist", page_icon="🔎", layout="wide")
init_db()

st.title("🔎 Screenshot Archaeologist")
st.caption("A local-first searchable archive for your screenshots. Images stay where they are; extracted text and metadata are stored in a local SQLite database.")

with st.sidebar:
    st.header("Index screenshots")
    folder = st.text_input("Folder path", placeholder=r"C:\Users\you\Pictures\Screenshots")
    st.caption("Start with a small test folder. OCR can take a while on large collections.")
    if st.button("Scan and index folder", type="primary", width="stretch"):
        if not folder.strip():
            st.error("Enter a folder path first.")
        elif not Path(folder).expanduser().exists():
            st.error("That folder does not exist on this computer.")
        else:
            with st.spinner("Scanning images and extracting text locally..."):
                result = index_folder(Path(folder).expanduser())
            st.success(f"Finished. {result['new']} new, {result['updated']} updated, {result['skipped']} skipped.")
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

    if st.button("Retry failed OCR", width="stretch"):
        with st.spinner("Retrying failed OCR for indexed screenshots..."):
            result = retry_failed_ocr()
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
    query = st.text_input("Search screenshots", placeholder="e.g. Python error, payment receipt")
    mode = st.radio("Search mode", ["Keyword (available now)", "Semantic (coming next)"], index=0)
    st.caption("Keyword search matches text found by OCR. Semantic search will be added after the core pipeline is verified.")

items = search_items(query) if query.strip() else list_items()
st.write(f"**{len(items)}** screenshot(s) {'matching your search' if query.strip() else 'indexed'}")

if not items:
    st.info("No screenshots to show yet. Enter a folder path in the sidebar and click **Scan and index folder**.")
else:
    cols_per_row = 3
    for start in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, item in zip(cols, items[start:start + cols_per_row]):
            with col:
                path = Path(item["path"])
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
                st.caption(f"{item['width']} × {item['height']} · {item['ocr_status']}")
                if item["ocr_text"]:
                    excerpt = item["ocr_text"].strip().replace("\n", " ")
                    st.write((excerpt[:240] + "…") if len(excerpt) > 240 else excerpt)
                else:
                    st.caption("No text extracted.")
                with st.expander("Details"):
                    st.code(item["path"])
                    st.write(f"Indexed: {item['indexed_at']}")
                    if item["ocr_text"]:
                        st.text_area("Extracted text", item["ocr_text"], height=160, key=f"text_{item['id']}")
                    if st.button("Open original file location", key=f"open_{item['id']}"):
                        st.info("Copy the path above into File Explorer. The app does not alter or delete your screenshots.")

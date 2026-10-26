# Screenshot Archaeologist

A local-first app that turns a folder of screenshots into a searchable archive. It currently provides folder import, OCR, SQLite indexing, keyword search, and a thumbnail review UI. Semantic search and local model categorization are intentionally the next milestone, not claimed as implemented yet.

## Privacy model

- Screenshot files are not uploaded.
- OCR runs locally through the installed Tesseract executable.
- Extracted text and file metadata are stored in `data/screenshots.db` beside the app.
- The app never deletes, renames, or modifies source screenshots.
- Treat the SQLite database as sensitive: it can contain text extracted from private images.

## Requirements

- Python 3.10+
- Tesseract OCR installed separately
- Windows, macOS, or Linux

### Install

1. Create and activate a virtual environment.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tesseract OCR:
   - Windows: install the Tesseract OCR Windows build and ensure `tesseract.exe` is on PATH.
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`
4. Start the app:
   ```bash
   streamlit run app.py
   ```

If Tesseract is missing, the app still indexes image metadata but reports OCR as unavailable.

## First run

Use a small folder of test screenshots first. In the sidebar, enter the folder path and click **Scan and index folder**. Search extracted text from the sidebar.

## Current limitations

- Keyword search only. Semantic search is the next milestone.
- OCR quality depends on image clarity, language packs, and Tesseract installation.
- The first scan reads every image in the selected folder recursively.
- The app currently displays up to 500 indexed records.
- The index stores absolute paths, so moving source files can break links until re-indexed.

## Roadmap

- [ ] Better scan progress and cancellation
- [ ] Duplicate detection UI
- [ ] Local semantic embeddings and ranking
- [ ] Local AI categories with evidence snippets
- [ ] Export/search diagnostics and tests

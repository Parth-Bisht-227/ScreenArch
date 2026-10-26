# Screenshot Archaeologist

A local-first app that turns a folder of screenshots into a searchable archive. It provides folder import, OCR, SQLite indexing, keyword search, a thumbnail review UI, and on-demand screenshot categorization with a locally running Ollama model.

## Privacy model

- Screenshot files are not uploaded.
- OCR runs locally through the installed Tesseract executable.
- Classification sends OCR text and basic file metadata only to the configured local Ollama endpoint. No image files or OCR text are sent to a cloud service.
- OCR text is treated as untrusted data; classification does not perform file operations.
- Extracted text and file metadata are stored in `data/screenshots.db` beside the app.
- Copy is the default organization operation and preserves source screenshots. Moving is available only as an explicitly selected and confirmed operation.
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
4. Install and start Ollama separately, then install a Gemma model in a terminal:
   ```bash
   ollama pull gemma3:4b
   ollama list
   ```
   Model downloads are explicit user actions; ScreenArch never downloads models. Ollama normally serves its local API at `http://localhost:11434`.
5. Start the app:
   ```bash
   streamlit run app.py
   ```

If Tesseract is missing, the app still indexes image metadata but reports OCR as unavailable.

## First run

Use a small folder of test screenshots first. In the sidebar, enter the folder path and click **Scan and index folder**. Search extracted text from the sidebar.

## Local AI organization

In **AI organization**, enter a natural-language instruction that explicitly lists the categories (for example, "Organize screenshots into Work, Personal, and Needs Review."). ScreenArch accepts those category labels only; it does not allow the model to invent new categories. Enter the exact installed Gemma model name shown by `ollama list`, check that the model is available, then run classification. The model name can also be set with `SCREENARCH_OLLAMA_MODEL`; it defaults to the suggested `gemma3:4b`, which must still be installed locally.

The default endpoint is `http://localhost:11434`. To use another local Ollama listener, set `SCREENARCH_OLLAMA_URL` to an `http://localhost`, `http://127.0.0.1`, or `http://[::1]` base URL. Non-local endpoints are rejected. Requests have timeouts and no cloud fallback.

The input limit controls how much OCR text (default 8,000 characters) is sent per screenshot; filename, dimensions, file size, and modified time are also sent. Screenshots with fewer than 20 alphanumeric characters or three OCR words are saved as **Needs Review** without a model call. Evidence must match a contiguous sequence of OCR words; case, whitespace, and punctuation differences are normalized, but paraphrases and invented words are rejected. Invalid evidence receives one focused correction attempt and is then sent to review if still invalid. Confidence is a model self-assessment from 0 to 1, not a calibrated probability; scores below 0.5 require review.

The batch UI reports completed/total items, the current filename, elapsed time, per-image status, actual model-request durations, and total batch duration. Result states distinguish insufficient OCR, invalid model output, invalid evidence, low confidence, and Ollama request failures. Failures include their cause and can be retried with **Reclassify unchanged items**.

### Review and organize

Classification never organizes files automatically. Edit categories and approve screenshots in **Review and approve**; **Approve all suitable results** only approves successful results that do not require review. Items categorized **Needs Review**, failed classifications, and unapproved items are not organized. Confidence alone never approves an item.

In **Organize approved screenshots**, choose a source folder in the sidebar and an output directory. The default output is `ScreenArch_Organized` inside the selected source folder, with one subfolder per approved category. **Copy (keep originals)** is the default. **Move (remove originals)** requires a separate confirmation and only removes each original after a verified copy. Existing files are never overwritten; duplicate names receive a numbered suffix. Completed operations are recorded in SQLite so repeating the same operation does not duplicate successful copies.

Classification, review decisions, and organization operation records are stored additively in the existing SQLite database; older OCR and classification rows remain usable.

## Current limitations

- Search is keyword-based; semantic search is not implemented.
- OCR quality depends on image clarity, language packs, and Tesseract installation.
- The first scan reads every image in the selected folder recursively.
- The app currently displays up to 500 indexed records.
- The index stores absolute paths, so moving source files can break links until re-indexed.
- Classification is text-only and depends on available OCR; it does not analyze image pixels.

## Roadmap

- [ ] Better scan progress and cancellation
- [ ] Duplicate detection UI
- [ ] Local semantic embeddings and ranking
- [x] Local AI categories with OCR evidence and review status
- [ ] Export/search diagnostics and tests

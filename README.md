# ScreenArch

**The archaeologist for your screenshots.**

A local-first searchable archive for your screenshots. ScreenArch indexes a local folder, extracts text with Tesseract OCR, and can classify screenshots from that text with a local Ollama model.

## Prerequisites

- Windows, macOS, or Linux.
- Python 3.10 or newer (64-bit recommended).
- Tesseract OCR installed separately from Python. On Windows, add the directory containing `tesseract.exe` to your `PATH`, then open a new terminal and verify with `tesseract --version`.
- Ollama installed and running locally for AI classification. The app uses a model already installed in Ollama; it does not download models automatically.
- A compatible local Ollama model, such as `gemma3:4b`, if you want AI classification. The selected model must be installed and its exact name must match `ollama list`.

The Python dependencies are listed in `requirements.txt`. In addition to those packages, the app uses Python's standard library and the pandas dependency provided by Streamlit. Tesseract and Ollama are separate applications and are not installed by pip.

## Install on Windows

Open PowerShell and run these commands. Replace the destination if you want the repository somewhere else.

```powershell
git clone https://github.com/Parth-Bisht-227/ScreenArch.git
Set-Location .\ScreenArch
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, allow scripts for just the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install Tesseract OCR separately using a Windows Tesseract installer. Ensure the folder containing `tesseract.exe` is on `PATH`, restart PowerShell, then check:

```powershell
tesseract --version
```

Install Ollama from [ollama.com/download](https://ollama.com/download/windows) and start the Ollama application. Confirm the command is available:

```powershell
ollama --version
```

If you choose `gemma3:4b`, download it explicitly and confirm the exact installed model name:

```powershell
ollama pull gemma3:4b
ollama list
```

Model downloads require connectivity. To use another model, install it with Ollama and use its exact name from `ollama list` in ScreenArch. The app does not fetch a model on your behalf.

From the activated virtual environment and the project directory, launch ScreenArch:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. If that port is in use, launch with another port, for example `streamlit run app.py --server.port 8502`.

## macOS and Linux

Clone the repository, create/activate a virtual environment, and install the same requirements:

```bash
git clone https://github.com/Parth-Bisht-227/ScreenArch.git
cd ScreenArch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Tesseract with your system package manager (for example, `brew install tesseract` on macOS or `sudo apt install tesseract-ocr` on Ubuntu/Debian), then verify with `tesseract --version`. Install Ollama from [ollama.com/download](https://ollama.com/download), start it, and check `ollama --version`. If you want Gemma 3 4B, run `ollama pull gemma3:4b` and `ollama list`. Start the app with `streamlit run app.py`.

## First run

1. In the sidebar, enter the path to a local screenshot folder. Start with a small test folder.
2. Select **Scan and index folder**. ScreenArch indexes image metadata and runs Tesseract OCR locally. Use **Retry failed OCR** after correcting an OCR installation problem.
3. Enter an organization instruction that explicitly names the categories. Comma-separated names or a numbered list with descriptions are supported, for example:

   ```text
   Organize screenshots into following categories:
   1. Content Inspirations [personal brand and growth]
   2. Engineering Resources [study resources]
   3. Job Help [career and job preparation]
   4. Miscellaneous [anything that does not clearly fit]
   ```

   Use **Needs Review** or **Miscellaneous** for uncertain screenshots; those results remain review-required when the model marks them uncertain.
4. Enter the exact installed Ollama model name shown by `ollama list` (for example, `gemma3:4b`). Select **Check local model** to verify the local service and model.
5. Select **Run classification** and follow the progress and per-image status. Use the category and review-status filters to inspect results.
6. In **Review and approve**, change categories if needed and save category/approval changes. Approve screenshots explicitly; **Approve all suitable results** excludes review-required results and the **Needs Review** category.
7. In **Organize approved screenshots**, choose an output directory. **Move (remove originals)** is selected by default and requires separate confirmation; choose **Copy (keep originals)** if you want to preserve source files. Only explicitly approved, non-review screenshots are organized.

The sidebar search performs keyword search over extracted OCR text. The selected source folder scopes the gallery, search, classification results, and organization actions; switching folders keeps each folder's cached OCR and classification results in SQLite without mixing them in the current view. Full extracted text is available in each screenshot's collapsed **View extracted text** expander.

## Troubleshooting

- **`tesseract` is not found:** Install Tesseract separately, add the directory containing `tesseract.exe` to the Windows `PATH`, open a new terminal, and check `tesseract --version`. In the app, select **Retry failed OCR** after fixing PATH. Without Tesseract, indexing can still record image metadata but OCR will be unavailable.
- **Ollama service unavailable:** Start the Ollama application/service, then try **Check local model** again. The default local API is `http://localhost:11434`; ScreenArch does not use a cloud fallback.
- **Model name not found:** Run `ollama list`, install the desired model with `ollama pull <model-name>` if necessary, and enter the exact name shown by the list. The app will not download it automatically.
- **Inference is slow:** Local inference time depends on the model, hardware, and screenshot count. Try a smaller compatible model or classify a smaller folder first; classification progress reports per-image status and request times.
- **A screenshot is marked `Needs Review`:** This is expected when OCR has fewer than 20 alphanumeric characters or fewer than three words, or when a model response/schema/category/evidence is invalid, uncertain, or below the review confidence threshold. Check the validation reason shown in the result, then adjust its category and approval manually. Validation logs include only the failing stage and a non-content reason code, not OCR or model response text.
- **Port conflict:** Use another port, for example `streamlit run app.py --server.port 8502`.
- **Virtual environment activation fails:** Confirm you are in the `ScreenArch` project directory. In PowerShell, use the process-scoped execution-policy command in the Windows setup section, then activate `.venv` again. On macOS/Linux, activate with `source .venv/bin/activate`.

## Privacy and limitations

- OCR runs locally through the installed Tesseract executable. Classification sends OCR text and basic file metadata to the configured local Ollama service. The model is not sent image pixels.
- The app has no cloud classification fallback. Initial package installation and Ollama model downloads require connectivity; do not assume every setup or update operation works offline.
- Search is keyword-based. Semantic search and vision-based classification are not implemented.
- Classification may be wrong. The confidence value is the model's self-assessment and is not calibrated; review and approve results before organizing files.
- **Move (remove originals)** is the default organization action and requires separate confirmation. A source is removed only after its copy is verified; its SQLite OCR and classification records are retained, but the old path is hidden from the active gallery. **Copy (keep originals)** remains available. The generated `ScreenArch_Organized` folder and known organization output directories are excluded from subsequent indexing.
- ScreenArch is a local application, not a hosted multi-user service. Extracted text, file metadata, review decisions, and organization records are stored in the local SQLite database at `data/screenshots.db`; protect that database as it may contain text from private images.

## Current limitations

- The first scan reads every image in the selected folder recursively.
- The app currently displays up to 500 indexed records.
- The index stores absolute paths, so moving source files can break links until they are re-indexed.

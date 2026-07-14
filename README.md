# HandwritingConverter

Turn photos of handwritten notes into searchable text and a printable PDF.

I built this because writing by hand is often the fastest way for me to think, but paper notes are difficult to search, revise, and print with my portable printer. HandwritingConverter connects those two workflows without pretending that handwriting OCR is perfect.

> **Status:** working beta. The full local conversion pipeline is implemented and covered by automated tests. Recognition quality still depends heavily on handwriting, lighting, camera angle, and the installed Tesseract language data.

## What it does

- accepts one or more JPG, PNG, TIFF, or other OpenCV-readable images
- preserves the image order as PDF page order
- resizes, denoises, increases contrast, and binarizes each page
- runs local Tesseract OCR without uploading notes to a cloud service
- writes a UTF-8 text transcription and a printable multipage PDF
- supports alternate Tesseract languages and page-segmentation modes
- can save processed images for recognition troubleshooting
- reports missing inputs and missing OCR installations clearly

## Demo

A real before/after screenshot is still needed. The repository includes sample images in `Input/` for local testing; they are not presented as accuracy benchmarks.

## Technology

- Python 3.11+
- OpenCV for image preparation
- Tesseract and pytesseract for OCR
- ReportLab for PDF generation
- pytest and Ruff for validation
- GitHub Actions for continuous integration

## Architecture

```text
handwriting image(s)
        |
        v
resize -> denoise -> contrast enhancement -> adaptive threshold
        |
        v
local Tesseract OCR
        |
        +----> UTF-8 transcription
        |
        +----> printable multipage PDF
```

## Installation

Tesseract is a system dependency and must be installed separately.

1. Install Python 3.11 or newer.
2. Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
3. Create and activate a virtual environment.
4. Install the project dependencies.

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If Tesseract is not on `PATH`, pass its executable explicitly:

```powershell
python main.py Input\test.jpg --tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Tesseract through your system package manager—for example, `brew install tesseract` on macOS or `sudo apt install tesseract-ocr` on Debian/Ubuntu.

## Usage

Convert a single page:

```bash
python main.py Input/test.jpg
```

Convert several pages in reading order:

```bash
python main.py page-1.jpg page-2.jpg page-3.jpg --output output/class-notes --title "Class notes"
```

This creates:

- `output/class-notes.txt`
- `output/class-notes.pdf`

Save the processed images when tuning recognition:

```bash
python main.py Input/test.jpg --debug-images --psm 6
```

Run `python main.py --help` for every option. Additional OCR languages require the corresponding Tesseract language package and can be selected with `--language`.

## Development

```bash
python -m pip install -r requirements-dev.txt
ruff check main.py tests
pytest -q
```

The tests isolate OCR itself so they remain deterministic, while still exercising preprocessing, multipage result handling, text output, and real PDF creation.

## Privacy

Images and recognized text stay on the local computer. This project does not send notes to an external OCR API.

## Known limitations

- Tesseract was designed primarily for printed text; highly cursive handwriting can produce poor results.
- Page rotation and perspective correction are not automatic yet.
- Text layout is intentionally simple and does not reproduce drawings or the exact page geometry.
- The committed sample photos may not represent every camera or handwriting style.

## Roadmap

- add automatic orientation and perspective correction
- compare local handwriting-focused OCR backends using the same sample set
- add a small desktop interface with page previews and editable text
- add documented before/after examples after manual accuracy review

## License and authorship

Created by [SimpleCaci](https://github.com/SimpleCaci) and released under the [MIT License](LICENSE).

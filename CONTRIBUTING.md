# Contributing to HandwritingConverter

Use Python 3.11+ and the Tesseract installation described in the README. Install `requirements-dev.txt` in a virtual environment, then run:

```bash
ruff check main.py tests
python -m pytest -q
```

The deterministic tests replace the OCR call while exercising image preparation, reports, and file output. Changes to recognition behavior should also include a local Tesseract run and identify the language data and page-segmentation mode used.

Keep sample notes synthetic or yours to share. Do not attach private class notes or someone else's handwriting without permission. For an OCR bug, describe lighting, image orientation, Tesseract version, command options, and expected versus observed text. Confidence scores are not accuracy measurements.

For a pull request, explain the change and validation. Include a report screenshot for presentation changes, and preserve the distinction between raw OCR and manually reviewed text. Discuss a new OCR backend or desktop interface in an issue before adding large dependencies.

Contributions follow the existing MIT license. There is no guaranteed support or response time.

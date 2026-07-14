"""Convert photographed handwriting into searchable text and reviewable documents."""

from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    text: str
    confidence: float | None = None
    processed_image: Path | None = None


def preprocess_image(path: Path, max_width: int = 2200):
    """Load, deskew-friendly enhance, and binarize a handwriting image."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unsupported or unreadable image: {path}")

    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    image = cv2.fastNlMeansDenoising(image, None, 12, 7, 21)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )


def verify_tesseract(command: str | None = None) -> None:
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    executable = pytesseract.pytesseract.tesseract_cmd
    if not shutil.which(executable) and not Path(executable).exists():
        raise RuntimeError(
            "Tesseract OCR was not found. Install Tesseract or pass --tesseract "
            "with the executable path."
        )
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract OCR is installed but could not be started.") from exc


def convert_images(
    inputs: Iterable[Path],
    *,
    language: str = "eng",
    page_segmentation: int = 6,
    debug_dir: Path | None = None,
) -> list[ConversionResult]:
    results: list[ConversionResult] = []
    config = f"--oem 3 --psm {page_segmentation}"

    for index, source in enumerate(inputs, start=1):
        if not source.is_file():
            raise FileNotFoundError(f"Input image does not exist: {source}")
        processed = preprocess_image(source)
        ocr_data = pytesseract.image_to_data(
            processed,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        words = [word.strip() for word in ocr_data["text"] if word.strip()]
        confidences = [
            float(value)
            for value in ocr_data["conf"]
            if float(value) >= 0
        ]
        text = " ".join(words)
        confidence = (
            sum(confidences) / len(confidences) if confidences else None
        )

        debug_path = None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{index:02d}-{source.stem}-processed.png"
            cv2.imwrite(str(debug_path), processed)

        results.append(
            ConversionResult(
                source=source,
                text=text,
                confidence=confidence,
                processed_image=debug_path,
            )
        )
    return results


def write_text(results: list[ConversionResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        f"# {item.source.name}\n\n{item.text or '[No text detected]'}"
        for item in results
    ]
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def write_pdf(results: list[ConversionResult], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.25 * inch)]
    for index, item in enumerate(results):
        story.extend(
            [
                Paragraph(item.source.name, styles["Heading2"]),
                Paragraph(
                    html.escape(item.text or "[No text detected]").replace("\n", "<br/>"),
                    styles["BodyText"],
                ),
            ]
        )
        if index < len(results) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(
        str(output),
        pagesize=letter,
        title=title,
        author="SimpleCaci",
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    ).build(story)


def _image_data_uri(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def write_review_report(
    results: list[ConversionResult], output: Path, title: str
) -> None:
    """Write a self-contained visual OCR review report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, item in enumerate(results, start=1):
        confidence = (
            "Confidence unavailable"
            if item.confidence is None
            else f"Average OCR confidence {item.confidence:.0f}%"
        )
        transcription = html.escape(item.text or "[No text detected]")
        cards.append(
            f"""
            <article class="page-card">
              <header>
                <span class="page-number">Page {index}</span>
                <h2>{html.escape(item.source.name)}</h2>
                <span class="confidence">{confidence}</span>
              </header>
              <div class="review-grid">
                <figure>
                  <img src="{_image_data_uri(item.source)}"
                       alt="Original handwriting page {index}">
                  <figcaption>Original page</figcaption>
                </figure>
                <section class="transcription" aria-label="OCR transcription">
                  <h3>Recognized text</h3>
                  <pre>{transcription}</pre>
                </section>
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} · OCR review</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #251f1a;
      --muted: #6f665f;
      --paper: #fffdf8;
      --line: #e5ddd0;
      --accent: #a75436;
      --wash: #f5eee4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #ebe4da;
      font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 48px auto; }}
    .intro {{ margin-bottom: 28px; }}
    .intro p {{ color: var(--muted); max-width: 68ch; }}
    .page-card {{
      margin: 24px 0;
      overflow: hidden;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 45px rgb(57 42 30 / 10%);
    }}
    .page-card > header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h2 {{ margin: 0; font-size: 1rem; flex: 1; }}
    .page-number, .confidence {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: .78rem;
      font-weight: 700;
    }}
    .page-number {{ color: white; background: var(--accent); }}
    .confidence {{ color: var(--accent); background: var(--wash); }}
    .review-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    figure {{ margin: 0; padding: 22px; border-right: 1px solid var(--line); }}
    img {{ width: 100%; max-height: 720px; object-fit: contain; background: white; border-radius: 10px; }}
    figcaption {{ color: var(--muted); font-size: .78rem; margin-top: 8px; }}
    .transcription {{ padding: 28px; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 1rem/1.65 ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
    @media (max-width: 760px) {{
      main {{ margin: 24px auto; }}
      .review-grid {{ grid-template-columns: 1fr; }}
      figure {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .page-card > header {{ flex-wrap: wrap; }}
      .confidence {{ width: 100%; }}
    }}
    @media print {{
      body {{ background: white; }}
      main {{ width: 100%; margin: 0; }}
      .page-card {{ box-shadow: none; break-after: page; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="intro">
      <h1>{html.escape(title)}</h1>
      <p>Compare every original page with its recognized text before editing,
      sharing, or printing the transcription.</p>
    </section>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn one or more handwriting photos into reviewable documents."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files in reading order")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/handwriting"))
    parser.add_argument("--title", default="Handwriting transcription")
    parser.add_argument("--language", default="eng", help="Installed Tesseract language code")
    parser.add_argument("--psm", type=int, default=6, choices=range(0, 14))
    parser.add_argument("--tesseract", help="Path to the Tesseract executable")
    parser.add_argument("--debug-images", action="store_true")
    parser.add_argument(
        "--no-review-report",
        action="store_true",
        help="Skip the self-contained HTML source/transcription review report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verify_tesseract(args.tesseract)
        results = convert_images(
            args.inputs,
            language=args.language,
            page_segmentation=args.psm,
            debug_dir=args.output.parent / "debug" if args.debug_images else None,
        )
        write_text(results, args.output.with_suffix(".txt"))
        write_pdf(results, args.output.with_suffix(".pdf"), args.title)
        if not args.no_review_report:
            write_review_report(results, args.output.with_suffix(".html"), args.title)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    detected = sum(bool(item.text) for item in results)
    print(f"Converted {len(results)} page(s); detected text on {detected}.")
    print(f"Text:   {args.output.with_suffix('.txt')}")
    print(f"PDF:    {args.output.with_suffix('.pdf')}")
    if not args.no_review_report:
        print(f"Review: {args.output.with_suffix('.html')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

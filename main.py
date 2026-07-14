"""Convert photographed handwriting into searchable text and a printable PDF."""

from __future__ import annotations

import argparse
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
        text = pytesseract.image_to_string(processed, lang=language, config=config).strip()

        debug_path = None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{index:02d}-{source.stem}-processed.png"
            cv2.imwrite(str(debug_path), processed)

        results.append(ConversionResult(source=source, text=text, processed_image=debug_path))
    return results


def write_text(results: list[ConversionResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = [f"# {item.source.name}\n\n{item.text or '[No text detected]'}" for item in results]
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
                    (item.text or "[No text detected]").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"),
                    styles["BodyText"],
                ),
            ]
        )
        if index < len(results) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(
        str(output), pagesize=letter, title=title, author="SimpleCaci",
        rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
    ).build(story)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn one or more handwriting photos into text and a printable PDF."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files in reading order")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/handwriting"))
    parser.add_argument("--title", default="Handwriting transcription")
    parser.add_argument("--language", default="eng", help="Installed Tesseract language code")
    parser.add_argument("--psm", type=int, default=6, choices=range(0, 14))
    parser.add_argument("--tesseract", help="Path to the Tesseract executable")
    parser.add_argument("--debug-images", action="store_true")
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    detected = sum(bool(item.text) for item in results)
    print(f"Converted {len(results)} page(s); detected text on {detected}.")
    print(f"Text: {args.output.with_suffix('.txt')}")
    print(f"PDF:  {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

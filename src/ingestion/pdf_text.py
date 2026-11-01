"""Page-level text extraction (PDF files and images): native text layer first, OCR only when it is missing.

A PDF page either carries real text (cheap, exact) or is just an image (needs
OCR, slower, and can misread characters). Choosing per page rather than per
file handles mixed documents, and recording which method produced each page
lets the database and the evaluation report how much of the corpus depended
on OCR.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import fitz  # PyMuPDF

from src.ingestion.ocr import ocr_image_bytes

MIN_NATIVE_CHARS = 20
MAX_PAGES = 60  # a longer file is refused: it would tie up the single OCR worker


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str
    method: str  # "native" or "ocr"


def extract_pages(path: pathlib.Path, ocr=ocr_image_bytes) -> list[PageText]:
    pages = []
    with fitz.open(path) as doc:
        if doc.page_count > MAX_PAGES:
            raise ValueError(f"the file has {doc.page_count} pages; the limit is {MAX_PAGES}")
        for index, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_NATIVE_CHARS:
                pages.append(PageText(index, text, "native"))
            else:
                # Render at ~130 dpi but never beyond ~2200 px on the long side (photos can be huge).
                zoom = min(130 / 72, 2200 / max(page.rect.width, page.rect.height))
                png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
                pages.append(PageText(index, ocr(png), "ocr"))
    return pages

"""OCR with RapidOCR (ONNX models, pip-installable, runs locally on CPU, free).

Tesseract would also work but needs a system install; RapidOCR keeps setup to
`pip install`. The engine is created lazily because loading it takes a moment
and most tests never need it.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_image_bytes(png: bytes) -> str:
    """Return recognised text, one line per detected text box, in reading order."""
    result, _ = _engine()(png)
    if not result:
        return ""
    # Each item is [box, text, confidence]; group boxes whose top edges are close into one line.
    boxes = sorted(result, key=lambda r: (r[0][0][1], r[0][0][0]))
    lines: list[list] = []
    for box in boxes:
        top = box[0][0][1]
        if lines and abs(top - lines[-1][0][0][0][1]) < 15:
            lines[-1].append(box)
        else:
            lines.append([box])
    return "\n".join(" ".join(b[1] for b in sorted(line, key=lambda r: r[0][0][0])) for line in lines)

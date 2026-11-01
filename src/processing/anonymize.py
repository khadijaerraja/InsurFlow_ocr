"""Pseudonymisation of personal data before text is stored.

The holder's name is replaced by a stable pseudonym (same person -> same
"Holder-AB12CD"), so records can still be grouped without exposing the name.
Emails and phone numbers are masked too. This is rule-based and only as good
as its patterns: it is a demonstration of the step, not a compliance tool, and
the original PDF is left untouched (here it is synthetic data anyway).
"""

from __future__ import annotations

import hashlib
import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?<!\d)(?:\+212|0)[\s.-]?[5-7](?:[\s.-]?\d{2}){4}(?!\d)")


def pseudonym(name: str) -> str:
    return "Holder-" + hashlib.sha256(name.strip().lower().encode()).hexdigest()[:6].upper()


def anonymize_text(text: str, names: list[str]) -> str:
    """Replace each name (and its space-less OCR variant) plus emails and phone numbers."""
    for name in {n.strip() for n in names if n and n.strip()}:
        parts = re.split(r"\s+", name)
        # "Amina El Idrissi" also matches "AminaElIdrissi" and "Amina  El Idrissi" (OCR drops/adds spaces).
        pattern = r"\s*".join(re.escape(p) for p in parts)
        text = re.sub(pattern, pseudonym(name), text, flags=re.I)
    text = _EMAIL.sub("[email removed]", text)
    return _PHONE.sub("[phone removed]", text)

"""Accent folding shared by keyword search in Python and in PostgreSQL.

French documents write "échéance" and users type "echeance" (or the reverse). Both sides are folded the same
way before matching: lower case, accents removed. PostgreSQL does it with `translate()` (immutable, so it can be
used in a generated column); Python does the same with the same character lists.
"""

from __future__ import annotations

ACCENTED = "àâäãåçéèêëíìîïñóòôöõúùûüýÿ"
PLAIN = "aaaaaceeeeiiiinooooouuuuyy"
assert len(ACCENTED) == len(PLAIN)
_TABLE = str.maketrans({**dict(zip(ACCENTED, PLAIN)), "œ": "oe", "æ": "ae"})


def fold(text: str) -> str:
    return text.lower().translate(_TABLE)

"""Rule-based classification and field extraction for four insurance document types, in English and French.

Rules are the first extractor on purpose: identifiers and labelled fields have a fixed shape, so rules are
exact, explainable and free. They are written for OCR output, which loses spaces ("Expirydate:2027-01-04")
and accents, so matching is done on accent-folded text with flexible separators between the letters of each
label, and guarantees/exclusions are matched against a vocabulary (English and French terms mapped to one
canonical name). Every value is normalised (ISO dates, float amounts, canonical identifiers) before being
stored or compared.

Dates are read as ISO (2027-01-04), day-first (04/01/2027) or spelled out (4 janvier 2027); amounts accept
"5 400,00 MAD", "1,250.50 MAD" and "DH".

Limits, stated plainly: a layout that uses other labels needs new entries in LABELS, or the grounded
language-model fallback (processing/llm_extract.py), which only fills what these rules could not find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from src.retrieval.text import ACCENTED, PLAIN

DOC_TYPES = ("contract", "amendment", "claim_form", "invoice")
_FOLD = str.maketrans(dict(zip(ACCENTED, PLAIN)))


def fold1(text: str) -> str:
    """Lower case and remove accents, character for character (so positions in the folded text are valid in the original)."""
    return text.lower().translate(_FOLD)


# ------------------------------------------------------------------------------------------ vocabulary
LABELS = {  # field -> label wordings (accent-free, lower case)
    "start": ["effective date", "date d effet", "date effet", "prise d effet", "debut de validite"],
    "end": ["expiry date", "date d echeance", "date echeance", "date d expiration", "fin de validite", "echeance"],
    "issued": ["issued on", "amendment date", "date of care", "invoice date", "etabli le", "date de l avenant",
               "date des soins", "date de facture"],
    "premium": ["annual premium", "prime annuelle"],
    "adjustment": ["premium adjustment", "ajustement de prime"],
    "claimed": ["claimed amount", "montant declare", "montant reclame"],
    "total": ["total due", "total a payer", "net a payer", "montant total"],
    "ceiling": ["annual ceiling", "plafond annuel", "plafond de remboursement"],
    "holder": ["policyholder", "patient", "claimant", "assure", "declarant", "beneficiaire", "titulaire"],
    "provider": ["provider", "etablissement de soins", "etablissement", "prestataire"],
}
_AMOUNT_FIELD = {"contract": "premium", "amendment": "adjustment", "claim_form": "claimed", "invoice": "total"}

_KEYWORDS = {
    "contract": ["health insurance contract", "contrat d assurance", "annual ceiling", "plafond annuel", "coverage",
                 "garanties", "exclusions"],
    "amendment": ["contract amendment", "amendment no", "premium adjustment", "avenant", "ajustement de prime"],
    "claim_form": ["claim declaration", "claim no", "claimed amount", "declaration de sinistre", "montant declare"],
    "invoice": ["medical invoice", "invoice no", "total due", "facture medicale", "total a payer"],
}

# canonical English name -> squashed (accent-free, letters and digits only) wordings, English and French
GUARANTEE_TERMS = {
    "Hospitalisation": ("hospitalisation",),
    "General practitioner": ("generalpractitioner", "medecingeneraliste"),
    "Outpatient care": ("outpatientcare", "soinsambulatoires"),
    "Pharmacy": ("pharmacy", "pharmacie"),
    "Dental care": ("dentalcare", "soinsdentaires"),
    "Maternity": ("maternity", "maternite"),
    "Optical care": ("opticalcare", "optique"),
}
EXCLUSION_TERMS = {
    "Cosmetic surgery": ("cosmeticsurgery", "chirurgieesthetique"),
    "Experimental treatment": ("experimentaltreatment", "traitementexperimental"),
    "Self-inflicted injury": ("selfinflictedinjury", "blessureautoinfligee"),
    "War and civil unrest": ("warandcivilunrest", "guerreettroublescivils"),
    "Alcohol or drug related injury": ("alcoholordrugrelatedinjury", "alcoolouauxdrogues"),
    "Pre-existing conditions (first 12 months)": ("preexistingconditions", "affectionspreexistantes"),
}

_REQUIRED = {
    "contract": ["policy_number", "start_date", "end_date", "annual_ceiling", "amount", "guarantees", "exclusions"],
    "amendment": ["policy_number", "reference", "start_date", "annual_ceiling", "amount"],
    "claim_form": ["policy_number", "reference", "document_date", "amount"],
    "invoice": ["policy_number", "reference", "document_date", "amount"],
}

_MONTHS = {m: i for i, names in enumerate(
    ["janvier january", "fevrier february", "mars march", "avril april", "mai may", "juin june", "juillet july",
     "aout august", "septembre september", "octobre october", "novembre november", "decembre december"], 1)
    for m in names.split()}
_DATE = (r"(?:(?P<y1>\d{4})\s*[-/.]\s*(?P<m1>\d{1,2})\s*[-/.]\s*(?P<d1>\d{1,2})"          # 2027-01-04
         r"|(?P<d2>\d{1,2})\s*[-/.]\s*(?P<m2>\d{1,2})\s*[-/.]\s*(?P<y2>\d{4})"            # 04/01/2027 (day first)
         r"|(?P<d3>\d{1,2})(?:er)?\s+(?P<mn>[a-z]{3,9})\s+(?P<y3>\d{4}))")                # 4 janvier 2027
_DATE_RE = re.compile(_DATE)
_AMOUNT = r"(\d[\d   ,]*(?:[.,]\d{2})?)\s*(?:mad|dhs?\b|dirhams?)"
_POLICY_RE = re.compile(r"\b(?:(INS)[-\s]?(\d{3,6})|P\s*[O0]\s*L\s*[-–]?\s*(\d{4})\s*[-–]?\s*(\d{6}))\b", re.I)
_REF_RE = re.compile(r"\b(SIN|FAC|AVN|CLM|INV)\s*[-–]?\s*(\d{3,6})\b", re.I)


@dataclass
class ExtractedDocument:
    doc_type: str = "unknown"
    policy_number: str | None = None
    reference: str | None = None
    amount: float | None = None
    start_date: str | None = None
    end_date: str | None = None
    document_date: str | None = None
    annual_ceiling: float | None = None
    holder: str | None = None
    provider: str | None = None
    guarantees: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)  # extraction/consistency problems found in this text

    def as_fields(self) -> dict:
        """Values worth storing (the holder's name is deliberately not among them: see anonymize.py)."""
        return {
            "policy_number": self.policy_number, "reference": self.reference, "amount": self.amount,
            "start_date": self.start_date, "end_date": self.end_date, "document_date": self.document_date,
            "annual_ceiling": self.annual_ceiling, "provider": self.provider,
            "guarantees": self.guarantees, "exclusions": self.exclusions, "dates": self.dates,
        }


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", fold1(text))


def _spaced(letters: str) -> str:
    """'expirydate' -> regex tolerant to missing spaces and to OCR reading a space as '.', '_' or '-'."""
    return r"[\s._'-]*".join(re.findall(r"[a-z]", letters))


def _label(variants: list[str]) -> str:
    """Regex matching any wording of a label, with flexible separators between its letters (longest first)."""
    ordered = sorted(variants, key=len, reverse=True)
    return "(?:" + "|".join(_spaced(re.sub(r"[^a-z]", "", v)) for v in ordered) + ")"


# -------------------------------------------------------------------------------------- scanned text
FIELD_LABELS = ["Contract No", "Amendment No", "Claim No", "Invoice No", "Policyholder", "Claimant", "Patient", "Provider",
                "Effective date", "Expiry date", "Issued on", "Amendment date", "Date of care", "Invoice date",
                "Annual premium", "New annual ceiling", "Annual ceiling", "Premium adjustment", "Claimed amount",
                "Total due", "Date d'effet", "Date d'échéance", "Prime annuelle", "Plafond annuel",
                "Nouveau plafond annuel", "Montant déclaré", "Total à payer", "Date des soins", "Date de facture"]


def normalize_labels(text: str) -> str:
    """Put the spaces back in known field labels of scanned text ("Claimedamount:" -> "Claimed amount:").

    OCR often glues small words together; restoring the labels keeps the stored passages readable and
    searchable. Only the labels this extractor knows are touched; matching ignores accents."""
    labels = sorted(FIELD_LABELS, key=len, reverse=True)  # longest first: "New annual ceiling" before "Annual ceiling"
    pattern = "|".join(f"(?P<g{i}>{_spaced(re.sub('[^a-z]', '', fold1(label)))})" for i, label in enumerate(labels))
    folded, out, last = fold1(text), [], 0
    for m in re.finditer(f"(?:{pattern})(?=\\s*:)", folded):
        out += [text[last:m.start()], labels[int(m.lastgroup[1:])]]
        last = m.end()
    return "".join(out) + text[last:]


def classify(text: str) -> str:
    """Keyword vote on text with spaces removed (OCR sometimes drops them) and accents folded."""
    squashed = _squash(text)
    scores = {t: sum(_squash(k) in squashed for k in kws) for t, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


# ------------------------------------------------------------------------------------ value parsers
def _parse_amount(raw: str) -> float:
    cleaned = re.sub(r"[\s  ]", "", raw)
    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    return float(cleaned)


def _date_from(m: re.Match) -> str | None:
    g = m.groupdict()
    try:
        if g["y1"]:
            return date(int(g["y1"]), int(g["m1"]), int(g["d1"])).isoformat()
        if g["y2"]:
            return date(int(g["y2"]), int(g["m2"]), int(g["d2"])).isoformat()
        month = _MONTHS.get(g["mn"])
        return date(int(g["y3"]), month, int(g["d3"])).isoformat() if month else None
    except ValueError:
        return None


def dates_in(folded: str) -> list[str]:
    seen: list[str] = []
    for m in _DATE_RE.finditer(folded):
        iso = _date_from(m)
        if iso and iso not in seen:
            seen.append(iso)
    return seen


def amounts_in(folded: str) -> set[float]:
    return {_parse_amount(m.group(1)) for m in re.finditer(_AMOUNT, folded)}


def _labelled_date(folded: str, variants: list[str]) -> str | None:
    m = re.search(_label(variants) + r"\W*" + _DATE, folded)
    return _date_from(m) if m else None


def _labelled_amount(folded: str, variants: list[str]) -> float | None:
    m = re.search(_label(variants) + r"\W*" + _AMOUNT, folded)
    return _parse_amount(m.group(1)) if m else None


def _value_after(text: str, folded: str, variants: list[str]) -> str | None:
    """Text following a 'Label:' at the start of a line (value on the same or the next line), original spelling."""
    m = re.search(r"(?m)^[ \t]*" + _label(variants) + r"[ \t]*:\s*([^\n]+)", folded)
    return text[m.start(1):m.end(1)].strip() if m else None


def _vocabulary(squashed: str, terms: dict[str, tuple[str, ...]]) -> list[str]:
    return [name for name, wordings in terms.items() if any(w in squashed for w in wordings)]


# ------------------------------------------------------------------------------------------ extract
def extract(text: str) -> ExtractedDocument:
    folded = fold1(text)
    doc = ExtractedDocument(doc_type=classify(text))

    if m := _POLICY_RE.search(text):
        doc.policy_number = f"INS-{int(m.group(2)):03d}" if m.group(1) else f"POL-{m.group(3)}-{m.group(4)}"
    if m := _REF_RE.search(text):
        doc.reference = f"{m.group(1).upper()}-{int(m.group(2)):03d}"
    if doc.doc_type == "contract" and doc.policy_number:
        doc.reference = doc.policy_number  # a contract is identified by its own number

    doc.start_date = _labelled_date(folded, LABELS["start"])
    doc.end_date = _labelled_date(folded, LABELS["end"])
    doc.document_date = _labelled_date(folded, LABELS["issued"])
    doc.dates = dates_in(folded)

    field_name = _AMOUNT_FIELD.get(doc.doc_type)
    doc.amount = _labelled_amount(folded, LABELS[field_name]) if field_name else None
    if doc.amount is None and (m := re.search(_AMOUNT, folded)):
        doc.amount = _parse_amount(m.group(1))
    doc.annual_ceiling = _labelled_amount(folded, LABELS["ceiling"])

    doc.holder = _value_after(text, folded, LABELS["holder"])
    doc.provider = _value_after(text, folded, LABELS["provider"])

    if doc.doc_type == "contract":
        squashed = _squash(text)
        cut = squashed.find("exclusions")
        covered, excluded = (squashed[:cut], squashed[cut:]) if cut != -1 else (squashed, "")
        doc.guarantees = _vocabulary(covered, GUARANTEE_TERMS)
        doc.exclusions = _vocabulary(excluded, EXCLUSION_TERMS)

    doc.issues = check(doc)
    return doc


def check(doc: ExtractedDocument) -> list[str]:
    """Missing or inconsistent values in what was extracted from this document alone."""
    if doc.doc_type == "unknown":
        return ["unclassified document"]
    issues = [f"missing {name.replace('_', ' ')}" for name in _REQUIRED[doc.doc_type] if not getattr(doc, name)]
    if doc.amount is not None and doc.amount <= 0:
        issues.append("non-positive amount")
    if doc.start_date and doc.end_date and doc.end_date <= doc.start_date:
        issues.append("expiry date is not after effective date")
    return issues

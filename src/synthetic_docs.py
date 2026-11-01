"""Generate synthetic, branded insurance PDFs (contracts, amendments, claims, invoices) with ground truth.

Real insurance documents cannot be published, so every document is built from
seeded values for a fictional insurer. The values used to render a document are
saved as ground truth (data/documents/ground_truth.json); that is what lets
extraction and retrieval be scored instead of eyeballed.

The set is deliberately not perfect: one claim points to a contract that was
never imported and one claim is dated before its contract started, so the
validation step has something real to catch.

Each document is written twice: a clean PDF with a text layer, and a "scanned"
copy that is only an image (rotated, blurred, noisy), so the OCR fallback has
realistic input. The `inbox` folder mixes both, like a real mailbox would.
"""

from __future__ import annotations

import io
import json
import pathlib
import random
from datetime import date, timedelta

import fitz  # PyMuPDF
from PIL import Image, ImageFilter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "documents"
POLICY_DIR = ROOT / "data" / "policies"

DOC_TYPES = ["contract", "amendment", "claim_form", "invoice"]
ISSUER = "Meridian Health Insurance"

NAVY, TEAL = HexColor("#17324D"), HexColor("#167D86")
SURFACE, BORDER, INK = HexColor("#F4F7FA"), HexColor("#D9E2EA"), HexColor("#243447")
MUTED, DANGER = HexColor("#5B6B7A"), HexColor("#A33A3A")

HOLDERS = ["Amina El Idrissi", "Youssef Benali", "Salma Tazi", "Omar Alaoui", "Nadia Berrada"]
ORPHAN_HOLDER = "Hamza Chraibi"
FR_HOLDER = "Karim Fassi"
PROVIDERS = {  # name -> brand colour used on the invoice
    "Clinique Atlas Marrakech": "#1F6FB2",
    "Polyclinique du Sud": "#B5532A",
    "Centre Medical Gueliz": "#2E7D5B",
    "Hopital Al Madina": "#8A2F4F",
}
CARE = ["Specialist consultation", "Hospital stay (2 nights)", "Dental treatment", "Laboratory tests",
        "Physiotherapy sessions"]

CEILINGS = {"Bronze": 50_000, "Silver": 150_000, "Gold": 500_000, "Platinum": 1_000_000}
PREMIUMS = {"Bronze": 1_800, "Silver": 3_200, "Gold": 5_400, "Platinum": 8_600}
GUARANTEES = {
    "Bronze": [("Hospitalisation", "70% of eligible costs"), ("General practitioner", "80% of eligible costs")],
    "Silver": [("Hospitalisation", "80% of eligible costs"), ("General practitioner", "90% of eligible costs"),
               ("Outpatient care", "up to 3 000 MAD per year"), ("Pharmacy", "70% of eligible costs")],
    "Gold": [("Hospitalisation", "100% of eligible costs"), ("General practitioner", "100% of eligible costs"),
             ("Outpatient care", "up to 5 000 MAD per year"), ("Pharmacy", "80% of eligible costs"),
             ("Dental care", "comprehensive"), ("Maternity", "after 12 months of cover")],
    "Platinum": [("Hospitalisation", "100% of eligible costs"), ("General practitioner", "100% of eligible costs"),
                 ("Outpatient care", "up to 10 000 MAD per year"), ("Pharmacy", "100% of eligible costs"),
                 ("Dental care", "comprehensive"), ("Maternity", "after 6 months of cover"),
                 ("Optical care", "up to 2 500 MAD per year")],
}
EXCLUSIONS = ["Cosmetic surgery", "Experimental treatment", "Self-inflicted injury", "War and civil unrest",
              "Alcohol or drug related injury", "Pre-existing conditions (first 12 months)"]

# French variants of the demo documents: other labels, day-first dates, decimal comma, "DH" on invoices.
FR_GUARANTEES = {"Hospitalisation": "Hospitalisation", "General practitioner": "Médecin généraliste",
                 "Outpatient care": "Soins ambulatoires", "Pharmacy": "Pharmacie", "Dental care": "Soins dentaires",
                 "Maternity": "Maternité", "Optical care": "Optique"}
FR_TERMS = {"100% of eligible costs": "100 % des frais éligibles", "80% of eligible costs": "80 % des frais éligibles",
            "90% of eligible costs": "90 % des frais éligibles", "70% of eligible costs": "70 % des frais éligibles",
            "comprehensive": "prise en charge complète", "after 12 months of cover": "après 12 mois de couverture",
            "after 6 months of cover": "après 6 mois de couverture"}
FR_EXCLUSIONS = {"Cosmetic surgery": "Chirurgie esthétique", "Experimental treatment": "Traitement expérimental",
                 "Self-inflicted injury": "Blessure auto-infligée", "War and civil unrest": "Guerre et troubles civils",
                 "Alcohol or drug related injury": "Blessure liée à l'alcool ou aux drogues",
                 "Pre-existing conditions (first 12 months)": "Affections préexistantes (12 premiers mois)"}


def _fmt_amount(value: float, locale: str = "en", currency: str = "MAD") -> str:
    text = f"{value:,.2f}".replace(",", " ")
    return (text.replace(".", ",") if locale == "fr" else text) + f" {currency}"


def _fr_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def build_documents(seed: int = 7) -> list[dict]:
    """Ground truth for the whole demo set, in delivery order."""
    rng = random.Random(seed)
    plans = sorted(p.stem for p in POLICY_DIR.glob("*.md")) or ["gold_family"]
    docs: list[dict] = []
    claim_no = 0
    for h, holder in enumerate(HOLDERS):
        plan = plans[(h * 7) % len(plans)]
        tier = plan.split("_")[0].title()
        contract_no = f"INS-{h + 1:03d}"
        start = date(2026, 1, 5) + timedelta(days=h * 17)
        end = start + timedelta(days=364)
        ceiling = float(CEILINGS[tier])
        base = {"holder": holder, "policy_number": contract_no, "product": plan.replace("_", " ").title()}
        docs.append({
            **base, "doc_type": "contract", "reference": contract_no, "amount": float(PREMIUMS[tier] + h * 50),
            "start_date": start.isoformat(), "end_date": end.isoformat(), "annual_ceiling": ceiling,
            "document_date": (start - timedelta(days=7)).isoformat(),
            "guarantees": [g for g, _ in GUARANTEES[tier]], "guarantee_terms": GUARANTEES[tier],
            "exclusions": sorted(rng.sample(EXCLUSIONS, 4), key=EXCLUSIONS.index), "anomaly": None,
        })
        if h in (0, 2):
            when = start + timedelta(days=90)
            docs.append({
                **base, "doc_type": "amendment", "reference": f"AVN-{h + 1:03d}",
                "amount": float(rng.choice([420, 650])), "start_date": when.isoformat(), "end_date": end.isoformat(),
                "annual_ceiling": ceiling + 50_000, "document_date": (when - timedelta(days=5)).isoformat(),
                "anomaly": None,
            })
        claim_no += 1
        provider = rng.choice(list(PROVIDERS))
        care_on = start + timedelta(days=rng.randint(30, 200))
        anomaly = None
        if h == 3:  # care dated before the contract started
            care_on, anomaly = start - timedelta(days=18), "outside_period"
        amount = round(rng.uniform(600, 9000), 2)
        care = rng.choice(CARE)
        docs.append({**base, "doc_type": "claim_form", "reference": f"SIN-{claim_no:03d}", "amount": amount,
                     "provider": provider, "care": care, "document_date": care_on.isoformat(), "anomaly": anomaly})
        docs.append({**base, "doc_type": "invoice", "reference": f"FAC-{claim_no:03d}", "amount": amount,
                     "provider": provider, "care": care, "document_date": care_on.isoformat(), "anomaly": anomaly})
    claim_no += 1
    docs.append({"holder": ORPHAN_HOLDER, "policy_number": "INS-099", "product": "Unknown", "doc_type": "claim_form",
                 "reference": f"SIN-{claim_no:03d}", "amount": 2350.0, "provider": "Polyclinique du Sud",
                 "care": "Specialist consultation", "document_date": "2026-05-12", "anomaly": "unknown_contract"})
    # A holder whose documents are in French (other labels, day-first dates, decimal comma, DH on the invoice).
    plan = "gold_individual" if "gold_individual" in plans else plans[0]
    start = date(2026, 3, 2)
    base = {"holder": FR_HOLDER, "policy_number": "INS-006", "product": plan.replace("_", " ").title(), "locale": "fr",
            "anomaly": None}
    docs.append({**base, "doc_type": "contract", "reference": "INS-006", "amount": 5_500.0,
                 "start_date": start.isoformat(), "end_date": (start + timedelta(days=364)).isoformat(),
                 "annual_ceiling": 500_000.0, "document_date": (start - timedelta(days=9)).isoformat(),
                 "guarantees": [g for g, _ in GUARANTEES["Gold"]], "guarantee_terms": GUARANTEES["Gold"],
                 "exclusions": ["Cosmetic surgery", "Alcohol or drug related injury", "Pre-existing conditions (first 12 months)"]})
    docs.append({**base, "doc_type": "claim_form", "reference": f"SIN-{claim_no + 1:03d}", "amount": 1_840.5,
                 "provider": "Hopital Al Madina", "care": "Séances de kinésithérapie", "document_date": "2026-07-09"})
    docs.append({**base, "doc_type": "invoice", "reference": f"FAC-{claim_no + 1:03d}", "amount": 1_840.5,
                 "provider": "Hopital Al Madina", "care": "Séances de kinésithérapie", "document_date": "2026-07-09"})
    return docs


def _layout_fr(t: dict) -> dict:
    """The French version of a document: other labels, day-first dates, decimal comma ("DH" on invoices)."""
    kind, d = t["doc_type"], _fr_date
    money = lambda v, cur="MAD": _fmt_amount(v, "fr", cur)  # noqa: E731
    sign = ("Signature autorisée", "Date et cachet")
    if kind == "contract":
        guarantees = [f"{FR_GUARANTEES[n]} - {FR_TERMS.get(terms, terms)}" for n, terms in t["guarantee_terms"]]
        return {
            "title": "CONTRAT D'ASSURANCE SANTÉ",
            "rows": [("N° de contrat", t["policy_number"]), ("Assuré", t["holder"]), ("Formule", t["product"]),
                     ("Date d'effet", d(t["start_date"])), ("Date d'échéance", d(t["end_date"])),
                     ("Prime annuelle", money(t["amount"])), ("Plafond annuel", money(t["annual_ceiling"])),
                     ("Établi le", d(t["document_date"]))],
            "sections": [("GARANTIES", guarantees), ("EXCLUSIONS", [FR_EXCLUSIONS[e] for e in t["exclusions"]])],
            "signature": True, "sign": sign,
        }
    if kind == "claim_form":
        return {
            "title": "DÉCLARATION DE SINISTRE",
            "rows": [("N° de sinistre", t["reference"]), ("N° de contrat", t["policy_number"]), ("Déclarant", t["holder"]),
                     ("Établissement", t["provider"]), ("Date des soins", d(t["document_date"])),
                     ("Montant déclaré", money(t["amount"]))],
            "sections": [("NATURE DES SOINS", [t["care"]])], "signature": True, "sign": sign,
        }
    return {
        "title": "FACTURE MÉDICALE",
        "rows": [("N° de facture", t["reference"]), ("N° de contrat", t["policy_number"]), ("Patient", t["holder"]),
                 ("Établissement", t["provider"]), ("Date de facture", d(t["document_date"])),
                 ("Total à payer", money(t["amount"], "DH"))],
        "sections": [("PRESTATIONS", [t["care"]])], "signature": False, "sign": sign,
    }


def _layout(t: dict) -> dict:
    """What a document says, independent of how it is drawn. Used for the PDF and the plain text."""
    if t.get("locale") == "fr":
        return _layout_fr(t)
    kind = t["doc_type"]
    common = [("Policyholder" if kind != "invoice" else "Patient", t["holder"])]
    if kind == "contract":
        return {
            "title": "HEALTH INSURANCE CONTRACT",
            "rows": [("Contract No", t["policy_number"]), *common, ("Plan", t["product"]),
                     ("Effective date", t["start_date"]), ("Expiry date", t["end_date"]),
                     ("Annual premium", _fmt_amount(t["amount"])),
                     ("Annual ceiling", _fmt_amount(t["annual_ceiling"])), ("Issued on", t["document_date"])],
            "sections": [("COVERAGE", [f"{n} - {terms}" for n, terms in t["guarantee_terms"]]),
                         ("EXCLUSIONS", list(t["exclusions"]))],
            "signature": True,
        }
    if kind == "amendment":
        return {
            "title": "CONTRACT AMENDMENT",
            "rows": [("Amendment No", t["reference"]), ("Contract No", t["policy_number"]), *common,
                     ("Amendment date", t["document_date"]), ("Effective date", t["start_date"]),
                     ("New annual ceiling", _fmt_amount(t["annual_ceiling"])),
                     ("Premium adjustment", _fmt_amount(t["amount"]))],
            "sections": [("REASON", ["Increase of the annual ceiling requested by the policyholder"])],
            "signature": True,
        }
    if kind == "claim_form":
        return {
            "title": "CLAIM DECLARATION",
            "rows": [("Claim No", t["reference"]), ("Contract No", t["policy_number"]), ("Claimant", t["holder"]),
                     ("Provider", t["provider"]), ("Date of care", t["document_date"]),
                     ("Claimed amount", _fmt_amount(t["amount"]))],
            "sections": [("DESCRIPTION OF CARE", [t["care"]])],
            "signature": True,
        }
    return {
        "title": "MEDICAL INVOICE",
        "rows": [("Invoice No", t["reference"]), ("Contract No", t["policy_number"]), *common,
                 ("Provider", t["provider"]), ("Invoice date", t["document_date"]),
                 ("Total due", _fmt_amount(t["amount"]))],
        "sections": [("SERVICES", [t["care"]])],
        "signature": False,
    }


def _lines(t: dict) -> list[str]:
    """Plain-text rendering of the same content (used by tests as a stand-in for OCR output)."""
    lay = _layout(t)
    out = [lay["title"]] + [f"{label}: {value}" for label, value in lay["rows"]]
    for heading, items in lay["sections"]:
        out.append(heading)
        out += [f"- {item}" for item in items]
    return out


def write_clean_pdf(t: dict, path: pathlib.Path) -> None:
    lay = _layout(t)
    is_invoice = t["doc_type"] == "invoice"
    brand = HexColor(PROVIDERS.get(t.get("provider", ""), "#17324D")) if is_invoice else NAVY
    issuer = t["provider"] if is_invoice else ISSUER
    width, height = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle(f"{lay['title'].title()} {t['reference']}")

    c.setFillColor(brand)
    c.rect(0, height - 92, width, 92, stroke=0, fill=1)
    c.setFillColor(TEAL if not is_invoice else NAVY)
    c.rect(0, height - 98, width, 6, stroke=0, fill=1)
    c.setFillColor(white)
    c.circle(62, height - 46, 19, stroke=0, fill=1)
    c.setFillColor(TEAL if not is_invoice else brand)
    c.circle(62, height - 46, 10, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(94, height - 44, issuer)
    c.setFont("Helvetica", 9)
    c.drawString(94, height - 60, "Marrakech, Morocco  |  Fictional demo company")

    y = height - 140
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 21)
    c.drawString(50, y, lay["title"])
    c.setStrokeColor(TEAL)
    c.setLineWidth(2.5)
    c.line(50, y - 9, 130, y - 9)

    y -= 40
    for i, (label, value) in enumerate(lay["rows"]):
        if i % 2 == 0:
            c.setFillColor(SURFACE)
            c.rect(50, y - 8, width - 100, 25, stroke=0, fill=1)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"{label}:")
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(220, y, value)
        y -= 26

    for heading, items in lay["sections"]:
        y -= 14
        danger = heading == "EXCLUSIONS"
        c.setFillColor(DANGER if danger else TEAL)
        c.rect(50, y - 6, width - 100, 22, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(60, y + 1, heading)
        y -= 26
        for item in items:
            c.setFillColor(DANGER if danger else TEAL)
            c.rect(60, y + 1, 6, 6, stroke=0, fill=1)
            name, _, terms = item.partition(" - ")
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold" if terms else "Helvetica", 10.5)
            c.drawString(76, y, name)
            if terms:
                c.setFont("Helvetica", 10.5)
                c.drawString(250, y, terms)
            c.setStrokeColor(BORDER)
            c.setLineWidth(0.5)
            c.line(60, y - 7, width - 60, y - 7)
            y -= 22

    if lay["signature"] and y > 140:
        c.setStrokeColor(MUTED)
        c.setLineWidth(0.8)
        c.line(60, 118, 240, 118)
        c.line(width - 240, 118, width - 60, 118)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8.5)
        c.drawString(60, 106, lay.get("sign", ("Authorised signature", "Date and stamp"))[0])
        c.drawString(width - 240, 106, lay.get("sign", ("Authorised signature", "Date and stamp"))[1])

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(50, 40, "Synthetic demonstration document - fictional data, no real person or insurer.")
    c.drawRightString(width - 50, 40, "Page 1 of 1")
    c.save()


def make_scanned_copy(clean_path: pathlib.Path, scanned_path: pathlib.Path, seed: int) -> None:
    """Rasterise a PDF and degrade it so it has no text layer (rotation, blur, noise)."""
    rng = random.Random(seed)
    with fitz.open(clean_path) as doc:
        pix = doc[0].get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    img = img.rotate(rng.uniform(-1.2, 1.2), fillcolor=(255, 255, 255), expand=False)
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    pixels = img.load()
    for _ in range(img.width * img.height // 150):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        v = rng.randint(150, 255)
        pixels[x, y] = (v, v, v)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    out = fitz.open()
    page = out.new_page(width=pix.width * 72 / 150, height=pix.height * 72 / 150)
    page.insert_image(page.rect, stream=buf.getvalue())
    out.save(scanned_path)
    out.close()


def generate(count: int | None = None, seed: int = 7, output_dir: pathlib.Path = OUTPUT_DIR) -> list[dict]:
    """Write the demo documents (clean + scanned versions) and ground_truth.json; return the truth list."""
    truths = build_documents(seed)[:count]
    for sub in ("clean", "scanned", "inbox"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
        for stale in (output_dir / sub).glob("*.pdf"):
            stale.unlink()  # a regenerated set must not mix with an older one
    for i, truth in enumerate(truths):
        name = f"{truth['doc_type']}_{i:02d}.pdf"
        clean, scanned = output_dir / "clean" / name, output_dir / "scanned" / name
        write_clean_pdf(truth, clean)
        make_scanned_copy(clean, scanned, seed=i)
        truth["file"] = name
        truth["scanned"] = i % 3 == 1  # about a third of the mailbox arrives as scans
        (output_dir / "inbox" / name).write_bytes((scanned if truth["scanned"] else clean).read_bytes())
    (output_dir / "ground_truth.json").write_text(json.dumps(truths, indent=2, ensure_ascii=False), encoding="utf-8")
    return truths


if __name__ == "__main__":
    print(f"Generated {len(generate())} documents in {OUTPUT_DIR}")

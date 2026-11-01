from src.processing.anonymize import anonymize_text, pseudonym
from src.processing.extraction import classify, extract

CONTRACT = """HEALTH INSURANCE CONTRACT
Contract No: INS-001
Policyholder: Amina El Idrissi
Effective date: 2026-01-05
Expiry date: 2027-01-04
Annual premium: 1 800.00 MAD
Annual ceiling: 50 000.00 MAD
COVERAGE
- Hospitalisation - 70% of eligible costs
- General practitioner - 80% of eligible costs
EXCLUSIONS
- Experimental treatment
- War and civil unrest
"""


def test_extracts_contract_fields_guarantees_and_exclusions():
    doc = extract(CONTRACT)
    assert doc.doc_type == "contract"
    assert doc.policy_number == doc.reference == "INS-001"
    assert (doc.start_date, doc.end_date) == ("2026-01-05", "2027-01-04")
    assert doc.amount == 1800.0 and doc.annual_ceiling == 50000.0
    assert doc.guarantees == ["Hospitalisation", "General practitioner"]
    assert doc.exclusions == ["Experimental treatment", "War and civil unrest"]
    assert doc.holder == "Amina El Idrissi"
    assert doc.issues == []


def test_claim_fields_keep_the_claim_and_the_contract_apart():
    doc = extract("CLAIM DECLARATION\nClaim No: SIN-003\nContract No: INS-001\nDate of care: 2026-03-14\n"
                  "Claimed amount: 1 250.50 MAD")
    assert doc.doc_type == "claim_form"
    assert (doc.reference, doc.policy_number) == ("SIN-003", "INS-001")
    assert doc.amount == 1250.50 and doc.document_date == "2026-03-14"


def test_tolerates_ocr_that_drops_spaces():
    doc = extract("MEDICALINVOICE\nInvoiceNo:FAC-004\nContractNo:INS-002\nInvoicedate:2026-05-02\nTotaldue:300.00MAD")
    assert doc.doc_type == "invoice"
    assert (doc.reference, doc.policy_number, doc.amount) == ("FAC-004", "INS-002", 300.0)
    assert doc.document_date == "2026-05-02"


def test_reports_missing_and_inconsistent_values():
    doc = extract("HEALTH INSURANCE CONTRACT\nContract No: INS-009\nEffective date: 2026-05-01\n"
                  "Expiry date: 2026-01-01\nAnnual premium: 100.00 MAD")
    assert "expiry date is not after effective date" in doc.issues
    assert "missing annual ceiling" in doc.issues and "missing guarantees" in doc.issues
    assert extract("some random text 2026-13-45").issues == ["unclassified document"]


def test_classify_survives_ocr_dropping_spaces():
    assert classify("CLAIMDECLARATION Claimedamount:5266.95MAD") == "claim_form"
    assert classify("CONTRACTAMENDMENT AmendmentNo:AVN-001") == "amendment"


def test_scanned_labels_get_their_spaces_back():
    from src.processing.extraction import normalize_labels

    fixed = normalize_labels("Claimedamount:4321.50MAD\nExpiry.date: 2027-01-01\nNewannualceiling:100000.00MAD")
    assert fixed.splitlines() == ["Claimed amount:4321.50MAD", "Expiry date: 2027-01-01", "New annual ceiling:100000.00MAD"]


def test_anonymisation_is_stable_and_handles_ocr_spacing():
    text = "Policyholder: Amina El Idrissi\nPatient AminaElIdrissi, mail amina@example.com, tel 0612345678"
    out = anonymize_text(text, ["Amina El Idrissi"])
    assert "Amina" not in out and "@" not in out and "0612345678" not in out
    assert out.count(pseudonym("Amina El Idrissi")) == 2
    assert pseudonym("amina el idrissi") == pseudonym("Amina El Idrissi")


# ------------------------------------------------------------------------------------------- French layouts
FRENCH_CONTRACT = """CONTRAT D'ASSURANCE SANTÉ
N° de contrat: INS-006
Assuré: Karim Fassi
Date d'effet: 02/03/2026
Date d'échéance: 1er mars 2027
Prime annuelle: 5 500,00 MAD
Plafond annuel: 500 000,00 MAD
GARANTIES
Hospitalisation - 100 % des frais éligibles
Médecin généraliste - 100 % des frais éligibles
Soins dentaires - prise en charge complète
EXCLUSIONS
Chirurgie esthétique
Affections préexistantes (12 premiers mois)
"""


def test_french_contract_is_read_with_day_first_dates_and_decimal_commas():
    doc = extract(FRENCH_CONTRACT)
    assert doc.doc_type == "contract" and doc.policy_number == "INS-006"
    assert (doc.start_date, doc.end_date) == ("2026-03-02", "2027-03-01")  # 02/03 is 2 March, "1er mars" spelled out
    assert doc.amount == 5500.0 and doc.annual_ceiling == 500000.0
    assert doc.guarantees == ["Hospitalisation", "General practitioner", "Dental care"]  # canonical English names
    assert doc.exclusions == ["Cosmetic surgery", "Pre-existing conditions (first 12 months)"]
    assert doc.holder == "Karim Fassi" and doc.issues == []


def test_french_invoice_in_dirhams_and_accent_free_ocr():
    doc = extract("FACTURE MEDICALE\nN. de facture: FAC-007\nN. de contrat: INS-006\nEtablissement: Hopital Al Madina\n"
                  "Date de facture: 09/07/2026\nTotal a payer: 1 840,50 DH")
    assert doc.doc_type == "invoice" and (doc.reference, doc.policy_number) == ("FAC-007", "INS-006")
    assert doc.amount == 1840.5 and doc.document_date == "2026-07-09" and doc.provider == "Hopital Al Madina"


def test_french_labels_are_restored_in_scanned_text():
    from src.processing.extraction import normalize_labels

    assert normalize_labels("Dated'echeance:01/03/2027\nMontantdeclare:1 840,50 MAD") == (
        "Date d'échéance:01/03/2027\nMontant déclaré:1 840,50 MAD")


# ------------------------------------------------------------------------------------ grounded model fallback
OTHER_LAYOUT = ("HEALTH CONTRACT\nPolicy reference INS-020\nCover starts on 15 March 2026 and ends on 14 March 2027.\n"
                "Yearly fee 4,200.00 MAD. Maximum reimbursement per year 250 000 MAD.\nCovered: hospitalisation and pharmacy.\n"
                "Not covered: cosmetic surgery.")


def test_fallback_fills_only_grounded_values_and_flags_them():
    import json

    from src.llm import ScriptedLLM
    from src.processing.llm_extract import make_fallback

    doc = extract(OTHER_LAYOUT)
    assert doc.start_date is None  # the rules do not know this wording
    answer = {"doc_type": "contract", "start_date": "2026-03-15", "end_date": "2027-03-14",
              "annual_ceiling": 250000, "amount": 4200.0, "policy_number": "INS-020",
              "document_date": "2020-01-01",          # invented: not in the text
              "provider": "Clinique Imaginaire"}      # invented: not in the text
    fill = make_fallback(ScriptedLLM([json.dumps(answer)]))
    filled = fill(OTHER_LAYOUT, doc)
    assert {"start_date", "end_date", "annual_ceiling"} <= set(filled)
    assert doc.end_date == "2027-03-14" and doc.annual_ceiling == 250000.0
    assert doc.document_date is None and doc.provider is None  # hallucinations are rejected


def test_fallback_never_overwrites_rule_results_and_skips_complete_documents():
    import json

    from src.llm import ScriptedLLM
    from src.processing.llm_extract import make_fallback

    doc = extract(CONTRACT)  # complete: the model must not even be called
    assert make_fallback(ScriptedLLM([]))(CONTRACT, doc) == []
    partial = extract("CLAIM DECLARATION\nClaim No: SIN-003\nContract No: INS-001\nClaimed amount: 100.00 MAD")
    assert partial.document_date is None and partial.amount == 100.0
    out = make_fallback(ScriptedLLM([json.dumps({"amount": 999.0, "document_date": "2026-05-01"})]))
    out("CLAIM DECLARATION\nClaim No: SIN-003\nContract No: INS-001\nClaimed amount: 100.00 MAD", partial)
    assert partial.amount == 100.0 and partial.document_date is None  # 999 and the date are not in the text


def test_fallback_survives_a_broken_model_answer():
    from src.llm import ScriptedLLM
    from src.processing.llm_extract import make_fallback

    doc = extract(OTHER_LAYOUT)
    assert make_fallback(ScriptedLLM(["I am sorry, I cannot do that."]))(OTHER_LAYOUT, doc) == []


def test_no_key_means_no_fallback(monkeypatch):
    from src.processing.llm_extract import make_fallback

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert make_fallback() is None

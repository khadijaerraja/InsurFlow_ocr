"""Generator tests: deterministic, in-range corpus size, no network involved."""

from src.generate_documents import TIERS, VARIANTS, generate_document


def test_generates_between_15_and_25_documents():
    count = len(TIERS) * len(VARIANTS)
    assert 15 <= count <= 25


def test_generate_document_is_deterministic():
    first = generate_document("Gold", "Family")
    second = generate_document("Gold", "Family")
    assert first == second


def test_generate_document_contains_expected_sections():
    text = generate_document("Bronze", "Senior")
    for heading in [
        "## Cover Levels",
        "## Exclusions",
        "## Waiting Periods",
        "## Claims Procedure",
        "## Geographic Scope",
    ]:
        assert heading in text


def test_tier_facts_differ_across_tiers():
    bronze = generate_document("Bronze", "Individual")
    platinum = generate_document("Platinum", "Individual")
    assert bronze != platinum
    assert "USD 50,000" in bronze
    assert "unlimited" in platinum

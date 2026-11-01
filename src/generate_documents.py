"""Synthesise a small insurance policy corpus with deterministic, tier-driven facts.

Real policy wordings cannot be used in a public portfolio repo, so this generator
builds a fictional product range (Bronze/Silver/Gold/Platinum, cross five customer
variants) from lookup tables rather than free-text templates. Keeping the numeric
facts (limits, waiting periods, claim deadlines) a deterministic function of
tier + variant -- rather than randomised -- means every fact in the gold
evaluation set (eval/questions.yaml) can be traced to exactly one document with
no ambiguity, which is what makes hit-rate/MRR scoring trustworthy.
"""

from __future__ import annotations

import pathlib
from typing import NamedTuple

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "policies"

TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
VARIANTS = ["Individual", "Family", "Senior", "Expatriate", "Student"]


class TierFacts(NamedTuple):
    inpatient_cover_pct: int
    outpatient_cover: str
    annual_limit: str
    dental: str
    maternity_waiting_months: int
    gp_waiting_days: int
    specialist_waiting_days: int
    pre_existing_rule: str
    claims_deadline_days: int
    claims_method: str


TIER_FACTS: dict[str, TierFacts] = {
    "Bronze": TierFacts(
        inpatient_cover_pct=70,
        outpatient_cover="not covered",
        annual_limit="USD 50,000",
        dental="not covered",
        maternity_waiting_months=24,
        gp_waiting_days=30,
        specialist_waiting_days=90,
        pre_existing_rule="excluded permanently unless declared and accepted at inception",
        claims_deadline_days=90,
        claims_method="reimbursement only: submit itemised receipts to the claims team",
    ),
    "Silver": TierFacts(
        inpatient_cover_pct=85,
        outpatient_cover="covered up to USD 2,000 per policy year",
        annual_limit="USD 150,000",
        dental="basic dental (check-ups, fillings) covered",
        maternity_waiting_months=18,
        gp_waiting_days=14,
        specialist_waiting_days=60,
        pre_existing_rule=(
            "excluded for the first 12 months unless declared and accepted at inception"
        ),
        claims_deadline_days=120,
        claims_method="reimbursement only: submit itemised receipts to the claims team",
    ),
    "Gold": TierFacts(
        inpatient_cover_pct=100,
        outpatient_cover="covered up to USD 5,000 per policy year",
        annual_limit="USD 500,000",
        dental="comprehensive dental (check-ups, fillings, root canal) covered",
        maternity_waiting_months=12,
        gp_waiting_days=0,
        specialist_waiting_days=30,
        pre_existing_rule="covered after 12 months of continuous cover on this policy",
        claims_deadline_days=180,
        claims_method="direct billing at network providers, or reimbursement with receipts",
    ),
    "Platinum": TierFacts(
        inpatient_cover_pct=100,
        outpatient_cover="covered with no annual cap",
        annual_limit="unlimited",
        dental="comprehensive dental and optical cover included",
        maternity_waiting_months=9,
        gp_waiting_days=0,
        specialist_waiting_days=0,
        pre_existing_rule="covered after 12 months of continuous cover on this policy",
        claims_deadline_days=180,
        claims_method="direct billing at network providers, or reimbursement with receipts",
    ),
}

VARIANT_GEOGRAPHY: dict[str, str] = {
    "Individual": (
        "Cover applies in the policyholder's country of residence, with emergency "
        "treatment covered worldwide for the first 30 days of any trip."
    ),
    "Family": (
        "Cover applies in the policyholder's country of residence for all named "
        "dependants, with emergency treatment covered worldwide for the first 30 "
        "days of any trip. Dependants are covered to age 18, or age 25 if in "
        "full-time education."
    ),
    "Senior": (
        "Cover applies worldwide excluding the United States, reflecting the "
        "higher cost of treatment there. Applicants aged 65 and over may be "
        "subject to an age-related premium loading at renewal."
    ),
    "Expatriate": (
        "Cover applies worldwide, including medical evacuation and repatriation "
        "to the policyholder's home country when locally unavailable treatment "
        "is required."
    ),
    "Student": (
        "Cover applies only within the policyholder's declared country of study, "
        "for the duration of the academic year stated on the policy schedule."
    ),
}

VARIANT_EXTRA_CLAUSE: dict[str, str] = {
    "Individual": "No additional clauses apply beyond the standard tier terms below.",
    "Family": (
        "Newborn children are automatically covered from birth for the first 30 "
        "days; continued cover requires notifying the insurer within that window."
    ),
    "Senior": (
        "Applicants must complete a health declaration at inception. Pre-existing "
        "conditions declared at inception are assessed individually and may carry "
        "a separate exclusion period beyond the standard rule below."
    ),
    "Expatriate": (
        "Security evacuation (civil unrest, natural disaster) is covered up to "
        "USD 100,000 per policy year in addition to medical evacuation."
    ),
    "Student": (
        "Cover ends automatically on course completion or after 12 months, "
        "whichever is sooner, unless renewed before expiry. Mental health "
        "support is included only on Gold and Platinum student policies."
    ),
}

COMMON_EXCLUSIONS = [
    "cosmetic or elective surgery not required for a diagnosed medical condition",
    "self-inflicted injury",
    "injury or illness arising from war, invasion, or civil unrest",
    "experimental or unlicensed treatment",
    "drugs or treatment not prescribed by a registered medical practitioner",
    "injury sustained while under the influence of alcohol or non-prescribed drugs",
]


def _policy_code(tier: str, variant: str) -> str:
    return f"{tier[:2].upper()}-{variant[:3].upper()}-2026"


def generate_document(tier: str, variant: str) -> str:
    """Render one policy document as Markdown for the given tier and variant."""
    facts = TIER_FACTS[tier]
    geography = VARIANT_GEOGRAPHY[variant]
    extra_clause = VARIANT_EXTRA_CLAUSE[variant]
    code = _policy_code(tier, variant)

    outpatient_line = (
        f"Outpatient treatment is {facts.outpatient_cover}."
        if facts.outpatient_cover == "not covered"
        else f"Outpatient treatment is {facts.outpatient_cover}."
    )
    gp_line = (
        "General practitioner (GP) consultations are covered from the policy "
        "start date with no waiting period."
        if facts.gp_waiting_days == 0
        else f"General practitioner (GP) consultations are subject to a "
        f"{facts.gp_waiting_days}-day waiting period from the policy start date."
    )
    specialist_line = (
        "Specialist consultations are covered from the policy start date with "
        "no waiting period."
        if facts.specialist_waiting_days == 0
        else f"Specialist consultations are subject to a "
        f"{facts.specialist_waiting_days}-day waiting period from the policy "
        f"start date."
    )

    return f"""# {tier} {variant} Health Policy

Policy code: {code}
Product line: Meridian Global Health -- {tier} tier, {variant} plan

## Cover Levels

Inpatient (hospital) treatment is covered at {facts.inpatient_cover_pct}% of
eligible costs. {outpatient_line} The annual claim limit for this policy is
{facts.annual_limit}. Dental cover: {facts.dental}.

## Exclusions

The following are excluded from all claims under this policy:

- {COMMON_EXCLUSIONS[0]}
- {COMMON_EXCLUSIONS[1]}
- {COMMON_EXCLUSIONS[2]}
- {COMMON_EXCLUSIONS[3]}
- {COMMON_EXCLUSIONS[4]}
- {COMMON_EXCLUSIONS[5]}

Pre-existing medical conditions are {facts.pre_existing_rule}.

## Waiting Periods

Maternity cover is subject to a {facts.maternity_waiting_months}-month waiting
period from the policy start date. {gp_line} {specialist_line}

## Claims Procedure

Claims must be submitted within {facts.claims_deadline_days} days of treatment.
Claims method: {facts.claims_method}.

## Geographic Scope

{geography}

## Additional Terms

{extra_clause}
"""


def generate_all(output_dir: pathlib.Path = OUTPUT_DIR) -> list[pathlib.Path]:
    """Write one Markdown file per tier x variant combination and return the paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tier in TIERS:
        for variant in VARIANTS:
            text = generate_document(tier, variant)
            filename = f"{tier.lower()}_{variant.lower()}.md"
            path = output_dir / filename
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


if __name__ == "__main__":
    paths = generate_all()
    print(f"Wrote {len(paths)} synthetic policy documents to {OUTPUT_DIR}")

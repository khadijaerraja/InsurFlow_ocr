"""Manual evaluation driver: builds both chunking strategies over the real corpus,
embeds with the local sentence-transformers model, and reports hit rate / MRR
for each against eval/questions.yaml.

Not part of `pytest` -- it downloads/loads the sentence-transformers model and
takes a few seconds to embed the corpus, so it is run by hand (or in CI only
if the model is pre-cached) rather than on every offline test run. The
numbers this script prints are the ones reported in the README.
"""

from __future__ import annotations

import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.chunking import Chunk, fixed_size_chunks, section_aware_chunks
from src.embeddings import embed_query, embed_texts
from src.llm import ScriptedLLM
from src.metrics import hit_rate, is_hit, mean_reciprocal_rank, reciprocal_rank
from src.qa import DEFAULT_CONFIDENCE_THRESHOLD, answer_question
from src.vector_store import VectorStore

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "policies"
QUESTIONS_PATH = pathlib.Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"
TOP_K = 5


def _load_corpus() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("*.md"))}


def _build_store(chunks: list[Chunk]) -> VectorStore:
    store = VectorStore()
    embeddings = embed_texts([c.text for c in chunks])
    store.add(chunks, embeddings)
    return store


def _evaluate(store: VectorStore, questions: list[dict]) -> dict:
    hits, ranks = [], []
    unanswerable_below_threshold = 0
    unanswerable_total = 0
    answerable_false_refusals = 0
    for q in questions:
        query_embedding = embed_query(q["question"])
        retrieved = store.search(query_embedding, top_k=TOP_K)
        top_score = retrieved[0].score if retrieved else 0.0
        if q["unanswerable"]:
            unanswerable_total += 1
            if top_score < DEFAULT_CONFIDENCE_THRESHOLD:
                unanswerable_below_threshold += 1
        else:
            hits.append(is_hit(retrieved, q["expected_doc"]))
            ranks.append(reciprocal_rank(retrieved, q["expected_doc"]))
            if top_score < DEFAULT_CONFIDENCE_THRESHOLD:
                answerable_false_refusals += 1
    return {
        "hit_rate": hit_rate(hits),
        "mrr": mean_reciprocal_rank(ranks),
        "n_answerable": len(hits),
        "unanswerable_below_threshold": unanswerable_below_threshold,
        "unanswerable_total": unanswerable_total,
        "answerable_false_refusals": answerable_false_refusals,
    }


def main() -> None:
    corpus = _load_corpus()
    questions = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(corpus)} documents, {len(questions)} gold questions\n")

    fixed_chunks = [c for name, text in corpus.items() for c in fixed_size_chunks(text, name)]
    section_chunks = [c for name, text in corpus.items() for c in section_aware_chunks(text, name)]
    print(f"Fixed-size chunking:    {len(fixed_chunks)} chunks")
    print(f"Section-aware chunking: {len(section_chunks)} chunks\n")

    fixed_store = _build_store(fixed_chunks)
    section_store = _build_store(section_chunks)

    fixed_results = _evaluate(fixed_store, questions)
    section_results = _evaluate(section_store, questions)

    print(f"=== Retrieval quality (top-{TOP_K}) ===")
    for name, results in [("fixed-size", fixed_results), ("section-aware", section_results)]:
        print(
            f"{name:>14}: hit_rate={results['hit_rate']:.3f}  mrr={results['mrr']:.3f}  "
            f"(n={results['n_answerable']} answerable questions)"
        )

    print(f"\n=== Hallucination guard (confidence threshold = {DEFAULT_CONFIDENCE_THRESHOLD}) ===")
    for name, results in [("fixed-size", fixed_results), ("section-aware", section_results)]:
        print(
            f"{name:>14}: {results['unanswerable_below_threshold']}/"
            f"{results['unanswerable_total']} unanswerable questions correctly refused; "
            f"{results['answerable_false_refusals']}/{results['n_answerable']} answerable "
            f"questions incorrectly refused (false refusals)"
        )

    print("\n=== Demo: end-to-end answer_question() with ScriptedLLM ===")
    grounded_llm = ScriptedLLM(
        ["Claims must be submitted within 120 days of treatment. [silver_family.md]"]
    )
    grounded = answer_question(
        "Within how many days must claims be submitted under a Silver Family policy?",
        section_store,
        embed_query,
        grounded_llm,
    )
    print(f"Grounded question -> refused={grounded.refused}, top_score={grounded.top_score:.3f}")
    print(f"  citations: {[(c.doc_id, c.section) for c in grounded.citations]}")

    refusal_llm = ScriptedLLM([])  # empty queue: raises if the guard fails to short-circuit
    refused = answer_question(
        "Does any Meridian Global Health policy cover flight delay compensation?",
        section_store,
        embed_query,
        refusal_llm,
    )
    print(f"Unanswerable question -> refused={refused.refused}, top_score={refused.top_score:.3f}")
    print(f"  answer: {refused.answer}")


if __name__ == "__main__":
    main()

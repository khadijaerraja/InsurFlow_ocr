"""Search scope shared by every retriever: one document, one contract, one document type.

The same object works as a Python predicate (in-memory search) and is translated to SQL conditions by the
PostgreSQL retriever, so both backends filter in exactly the same way.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Filter:
    document_id: int | None = None
    contract: str | None = None
    doc_type: str | None = None

    def __bool__(self) -> bool:
        return any(v is not None for v in (self.document_id, self.contract, self.doc_type))

    def __call__(self, chunk) -> bool:
        return ((self.document_id is None or chunk.document_id == self.document_id)
                and (self.contract is None or chunk.contract == self.contract)
                and (self.doc_type is None or chunk.section == self.doc_type))

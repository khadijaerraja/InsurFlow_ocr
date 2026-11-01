# InsurFlow documentation

This folder is the home for project documentation that complements the root
[README](../README.md). Documentation must describe the implementation that
exists today and label planned capabilities as planned rather than available.

## Current documents

- [Architecture and evolution](architecture.md): current implementation,
  mapping to proposed target modules, and a staged evolution path.
- [Brand guidelines](brand-guidelines.md): initial product identity, palette,
  typography, and interface principles.
- [Current architecture diagram](diagrams/current-architecture.mmd):
  implemented components and clearly marked planned layers.
- [Target architecture diagram](diagrams/target-architecture.mmd):
  proposed future platform, separate from the current implementation.
- [RAG workflow diagram](diagrams/rag-workflow.mmd): current retrieval,
  refusal, generation, and citation control flow.

## Documentation structure

As the project grows, keep focused documents here:

- `data-model.md`: implemented entities, relationships, and migration notes.
- `ingestion.md`: supported inputs and the actual extraction pipeline.
- `evaluation.md`: reproducible evaluation datasets, commands, and measured
  results.
- `security.md`: tested security controls and explicit limitations.
- `development.md`: local setup, supported commands, and contribution notes.

The future document names above are organizational suggestions, not claims
that those features or documents have already been implemented. Keep the
upstream MIT license and copyright notices intact when adapting or
redistributing covered code.

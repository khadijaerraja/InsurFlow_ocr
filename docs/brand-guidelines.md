# InsurFlow brand guidelines

## Product name

**InsurFlow — Intelligent Insurance Data Platform**

Use the full name on the project landing page and in portfolio descriptions.
Use **InsurFlow** in navigation, page titles, and compact labels. Keep existing
Python package names, classes, functions, and data identifiers unchanged
unless a later, separately tested change requires otherwise.

## Positioning

InsurFlow is the target identity for a local-first insurance document
intelligence portfolio project. The current implementation is a Python RAG
prototype over synthetic Markdown policies. PDF ingestion, OCR, a database,
authentication, and a frontend are not part of the current code.

## Visual direction

The intended interface is a restrained B2B SaaS product: clear hierarchy,
readable tables, useful status indicators, and a focus on evidence and
document provenance. Avoid ornamental gradients, decorative illustrations,
and data visualizations without a real data source.

### Initial palette

| Token | Hex | Intended use |
|---|---|---|
| `navy` | `#17324D` | Primary navigation, headings, and high-emphasis text |
| `teal` | `#167D86` | Primary actions and selected navigation state |
| `ink` | `#243447` | Body text |
| `surface` | `#F4F7FA` | Application background |
| `white` | `#FFFFFF` | Cards and content surfaces |
| `border` | `#D9E2EA` | Dividers, table borders, and input outlines |
| `success` | `#287A55` | Successful processing states |
| `warning` | `#A65C00` | Attention and review-needed states |
| `danger` | `#A33A3A` | Errors and failed processing states |

Use status text or icons alongside color; color alone must not convey meaning.
Check text and control contrast when implementing the interface.

### Architecture diagrams

Use the same light background, navy labels, teal implemented components, and
neutral gray planned components. Use solid borders for implemented work,
dashed borders for planned work, and a separate blue-gray treatment for
optional external integrations. Keep nodes aligned by system layer, use
directional arrows, and apply consistent modestly rounded corners. Icons
should be simple, monochrome glyphs used consistently rather than decorative
illustrations.

### Typography

Prefer a neutral system sans-serif stack for interface text and a
system-monospace stack for identifiers, policy codes, and technical metadata.
This keeps the initial visual direction dependency-free and locally available.

## Interface principles

- Keep navigation predictable and labels aligned with insurance workflows.
- Show document status, source, and confidence in plain language.
- Keep extracted values distinguishable from human-verified values.
- Make source citations and page references easy to inspect when supported.
- Use responsive layouts, keyboard-accessible controls, and visible focus.
- Do not display invented processing statistics or imply unimplemented
  security, OCR, or workflow capabilities.

## Attribution

The existing implementation is adapted from the `insurance-document-rag`
repository. Preserve its MIT license and upstream copyright notice in
[`../LICENSE`](../LICENSE), and keep the origin and scope statement in the
[project README](../README.md).

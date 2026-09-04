---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# ASM1 Canonical Knowledge Base

This vault records the ASM1 workbook without silently correcting source notation. Machine consumers must use [`data/asm1.json`](data/asm1.json); Markdown files are generated views of that JSON.

## Navigation

- [[State Variables]]
- [[Parameters]]
- [[Processes and Rates]]
- [[Corrected Matrices]]
- [[Composition and Continuity]]
- [[Kinetic Checking Matrix]]
- [[Source Manifest]]
- [[Audit Report]]

## Contract

- Operational component vector: **14** components (13 listed state variables plus matrix-only `SN2`).
- Processes: **8**.
- Parameters: **10 stoichiometric + 15 kinetic**.
- Source SHA-256: `dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a`.
- Source text has precedence over normalized aliases. `code_id` and `code_expression` are additive implementation aids.

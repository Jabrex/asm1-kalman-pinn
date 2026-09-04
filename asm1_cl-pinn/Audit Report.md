---
model: ASM1
audit_status: PASS
audit_date: 2026-08-27
auditor: asm1_final_independent_audit
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
json_sha256: 06f7bfd5ce5703f5745cc0565858fc147012e68b3e28fdca64a126e6ed7074a2
schema_sha256: c4b61018713b26ea76b7c149fb0708452691aae2fa2473a85ce9230d5a6459f3
---

# Audit Report

## Status

**PASS** — final independent source-to-vault mismatch count: **0**.

The independent subagent read `asm1.xlsx` afresh with artifact-tool and a separate OOXML inspection. It did not trust the vault generation logic. An initial audit found two silently trimmed trailing spaces in `Sheet1!E25` and `Sheet1!E27`; those defects were corrected, regression-tested, republished, and the complete independent audit was then repeated from the beginning. The result below applies to the corrected final vault.

## Cryptographic identity

| Artifact | SHA-256 |
| --- | --- |
| Source `asm1.xlsx` | `dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a` |
| Canonical `data/asm1.json` | `06f7bfd5ce5703f5745cc0565858fc147012e68b3e28fdca64a126e6ed7074a2` |
| Contract `data/asm1.schema.json` | `c4b61018713b26ea76b7c149fb0708452691aae2fa2473a85ce9230d5a6459f3` |

The source workbook was unchanged by the build and final audit. The generator and publish commands did not target `.obsidian`; four of its five configuration files retained their audited hashes. Because the vault was open in Obsidian, the application automatically rewrote the dynamic UI-state file `.obsidian/workspace.json` after `Audit Report.md` appeared. Its hash changed from `cfc51fefe0fdf3027c009cdd5d38452e2aca10c79f370d8b2ff264af6563f747` (5,616 bytes; `2026-08-27T17:52:41.2503186Z`) to `04ab747fda8eb2606db6f48ba147ed37ee4014f43528a9da24efab492d6bd57a` (5,230 bytes; `2026-08-27T18:14:13.9137561Z`). The updated `lastOpenFiles` contains `Audit Report.md`; this is an Obsidian workspace-state change, not an ASM1 data or model change. The prior bytes were not available, so no speculative restoration was attempted.

## Independent acceptance results

| Check | Result | Evidence |
| --- | --- | --- |
| Workbook scope | PASS | One worksheet; `Sheet1!A1:AC91` |
| Operational components | PASS | 13 listed states + matrix-only `SN2` = 14 |
| Parameters | PASS | 10 stoichiometric + 15 kinetic |
| Processes and rates | PASS | 8 processes; 16 source rate strings matched character-for-character |
| Corrected matrices | PASS | Original and standardised matrices are each `8x14` |
| Composition matrices | PASS | Two symbolic `3x14` matrices and one numeric `14x3` matrix |
| Numeric stoichiometry and continuity | PASS | `8x14` stoichiometry and `8x3` residual matrix |
| Kinetic checking | PASS | `8x14`; 13 consumed, 8 biomass, 4 other-required, 1 inhibitory |
| Missing alkalinity terms | PASS | `X82` and `X84` preserved as source omissions; no term invented |
| Formula manifest | PASS | 56 formula cells; calcChain count 56; Excel error tokens 0 |
| Named ranges | PASS | 10 |
| Legacy objects | PASS | 3 distinct zero-size `Equation.3` object identities; two unique payload hashes because objects 1 and 3 are byte-identical in the source |
| Schema | PASS | PowerShell `Test-Json=True`; every declared object schema uses `additionalProperties: false` |
| Markdown views | PASS | Eight generated views agree with JSON and source hash |
| Source anomalies | PASS | 14 records; `E25="YOHO "` and `E27="YANO "` retain their final U+0020 spaces |
| Vault configuration | EXTERNAL CHANGE | Four `.obsidian` files unchanged; open Obsidian rewrote dynamic `.obsidian/workspace.json` as documented above |

## Numerical and notation checks

- Corrected-matrix alias mismatches: **0**.
- Composition-matrix alias mismatches: **0**.
- Rate-expression alias mismatches: **0**.
- Source-to-code rate mismatches: **0**.
- Independently recomputed maximum absolute continuity residual: `5.551115123125783e-17`.
- Required tolerance: `<= 1e-15`; result: **PASS**.
- Corrected/numeric matrices preserve 81 source blanks as `raw_cell: null` with effective zero.
- Composition/continuity matrices preserve 21 explicit numeric zeros separately from blanks.
- `SNOx`/`SNHx`, `SNOX`/`SNHX`, and continuity aliases remain distinct source representations linked only through the explicit alias table.

## Literature cross-check

- [Hauduc et al. (2010)](https://pubmed.ncbi.nlm.nih.gov/20182061/) describes systematic verification of activated-sludge-model stoichiometry and kinetic expressions, with corrected matrices and continuity checks. The vault audit follows that verification structure.
- [Corominas et al. (2010)](https://publications.polymtl.ca/18466/) defines a systematic standardised notation framework for wastewater-treatment models. The vault therefore preserves original and standardised source symbols separately and adds explicit code-safe aliases without overwriting source text.

## Final declaration

The canonical JSON, schema, and generated Markdown views satisfy the requested ASM1 coverage and provenance requirements. The final independent data audit found no remaining source-to-vault differences. The only closing exception is the explicitly documented automatic Obsidian UI-state update; it does not affect canonical ASM1 content.

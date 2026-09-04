---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# State Variables

The source table lists 13 state variables. `SN2` is preserved as the fourteenth operational component because it is explicitly added to both corrected matrices.

| # | Original | State-table standardised | Matrix header | Continuity | Code ID | Description | Unit | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `SI` | `SU` | `SU` | `S_I` | `S_I` | Soluble undegradable organics | g COD.m-3 | listed | Sheet1!D13 |
| 2 | `SS` | `SB` | `SB` | `S_S` | `S_S` | Soluble biodegradable organics | g COD.m-3 | listed | Sheet1!D12 |
| 3 | `XI` | `XU,Inf` | `XU,Inf` | `X_I` | `X_I` | Particulate undegradable organics from the influent | g COD.m-3 | listed | Sheet1!D16 |
| 4 | `XS` | `XCB` | `XCB` | `X_S` | `X_S` | Particulate biodegradable organics | g COD.m-3 | listed | Sheet1!D15 |
| 5 | `XB,H` | `XOHO` | `XOHO` | `X_B.H` | `X_B_H` | Ordinary heterotrophic organisms | g COD.m-3 | listed | Sheet1!D22 |
| 6 | `XB,A` | `XANO` | `XANO` | `X_B.A` | `X_B_A` | Autotrophic nitrifying organisms (NH4+ to NO3-) | g COD.m-3 | listed | Sheet1!D23 |
| 7 | `XP` | `XU,E` | `XU,E` | `X_P` | `X_P` | Particulate undegradable endogenous products | g COD.m-3 | listed | Sheet1!D17 |
| 8 | `SO` | `SO2` | `SO2` | `S_O` | `S_O` | Dissolved oxygen | - g COD.m-3 | listed | Sheet1!D14 |
| 9 | `SNO` | `SNOx` | `SNOX` | `S_NO` | `S_NO` | Nitrate and nitrite (NO3 + NO2) (considered to be NO3 only for stoichiometry) | g N.m-3 | listed | Sheet1!D19 |
| 10 | `SNH` | `SNHx` | `SNHX` | `S_NH` | `S_NH` | Ammonia (NH4 + NH3) | g N.m-3 | listed | Sheet1!D18 |
| 11 | `SND` | `SB,N` | `SB,N` | `S_ND` | `S_ND` | Soluble biodegradable organic N | g N.m-3 | listed | Sheet1!D21 |
| 12 | `XND` | `XCB,N` | `XCB,N` | `X_ND` | `X_ND` | Particulate biodegradable organic N | g N.m-3 | listed | Sheet1!D20 |
| 13 | `SALK` | `SAlk` | `SAlk` | `S_ALK` | `S_ALK` | Alkalinity (HCO3-) | mol HCO3-.m-3 | listed | Sheet1!D24 |
| 14 | `SN2` | `SN2` | `SN2` | `S_N2` | `S_N2` | — | — | added_term_not_listed | Sheet1!Y11 |

---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# Composition and Continuity

## Original composition matrix

| Process / conserved quantity | SI | SS | XI | XS | XB,H | XB,A | XP | SO | SNO | SNH | SND | XND | SALK | SN2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COD | 1 | 1 | 1 | 1 | 1 | 1 | 1 | -1 | iCOD_NO3 | 0 | 0 | 0 | 0 | iCOD_N2 |
| N | 0 | 0 | 0 | 0 | iXB | iXB | iXP | 0 | 1 | 1 | 1 | 1 | 0 | 1 |
| Charge | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | iCharge_SNOx | iCharge_SNHx | 0 | 0 | -1 | 0 |

## Standardised composition matrix

| Process / conserved quantity | SU | SB | XU,Inf | XCB | XOHO | XANO | XU,E | SO2 | SNOX | SNHX | SB,N | XCB,N | SAlk | SN2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COD | 1 | 1 | 1 | 1 | 1 | 1 | 1 | -1 | iCOD_NO3 | 0 | 0 | 0 | 0 | iCOD_N2 |
| N | 0 | 0 | 0 | 0 | iN_XBio | iN_XBio | iN_XUE | 0 | 1 | 1 | 1 | 1 | 0 | 1 |
| Charge | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | iCharge_SNOx | iCharge_SNHx | 0 | 0 | -1 | 0 |

## Continuity composition matrix

| Process / conserved quantity | COD | N | Charge |
| --- | --- | --- | --- |
| S_I | 1 | 0 | 0 |
| S_S | 1 | 0 | 0 |
| X_I | 1 | 0 | 0 |
| X_S | 1 | 0 | 0 |
| X_B.H | 1 | 0.085999999999999993 | 0 |
| X_B.A | 1 | 0.085999999999999993 | 0 |
| X_P | 1 | 0.059999999999999998 | 0 |
| S_O | -1 | 0 | 0 |
| S_NO | -4.5714285714285712 | 1 | -0.071428571428571425 |
| S_NH | 0 | 1 | 0.071428571428571425 |
| S_ND | 0 | 1 | 0 |
| X_ND | 0 | 1 | 0 |
| S_ALK | 0 | 0 | -1 |
| S_N2 | -1.7142857142857142 | 1 | 0 |

## Numeric stoichiometry

| Process / conserved quantity | S_I | S_S | X_I | X_S | X_B.H | X_B.A | X_P | S_O | S_NO | S_NH | S_ND | X_ND | S_ALK | S_N2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aerobic growth of heterotrophs | ∅ (0) | -1.4925373134328357 | ∅ (0) | ∅ (0) | 1 | ∅ (0) | ∅ (0) | -0.49253731343283574 | ∅ (0) | -0.085999999999999993 | ∅ (0) | ∅ (0) | -0.0061428571428571417 | ∅ (0) |
| Anoxic growth of heterotrophs | ∅ (0) | -1.4925373134328357 | ∅ (0) | ∅ (0) | 1 | ∅ (0) | ∅ (0) | ∅ (0) | -0.17238805970149251 | -0.085999999999999993 | ∅ (0) | ∅ (0) | 0.0061705756929637508 | 0.17238805970149251 |
| Aerobic growth of autotrophs | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | ∅ (0) | -18.047619047619047 | 4.166666666666667 | -4.2526666666666673 | ∅ (0) | ∅ (0) | -0.60138095238095235 | ∅ (0) |
| Decay of heterotrophs | ∅ (0) | ∅ (0) | ∅ (0) | 0.92000000000000004 | -1 | ∅ (0) | 0.080000000000000002 | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 0.081199999999999994 | ∅ (0) | ∅ (0) |
| Decay of autotrophs | ∅ (0) | ∅ (0) | ∅ (0) | 0.92000000000000004 | ∅ (0) | -1 | 0.080000000000000002 | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 0.081199999999999994 | ∅ (0) | ∅ (0) |
| Ammonification of soluble organic Nitrogen | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | -1 | ∅ (0) | 0.071428571428571425 | ∅ (0) |
| Hydrolysis of entrapped organics | ∅ (0) | 1 | ∅ (0) | -1 | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) |
| Hydrolysis of entrapped organic nitrogen | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | -1 | ∅ (0) | ∅ (0) |

## Independently recomputed residuals

| # | Process | COD | N | Charge |
| --- | --- | --- | --- | --- |
| 1 | Aerobic growth of heterotrophs | 5.5511151231257827e-17 | 0 | 0 |
| 2 | Anoxic growth of heterotrophs | 0 | 0 | 0 |
| 3 | Aerobic growth of autotrophs | 0 | 0 | 0 |
| 4 | Decay of heterotrophs | 4.163336342344337e-17 | 0 | 0 |
| 5 | Decay of autotrophs | 4.163336342344337e-17 | 0 | 0 |
| 6 | Ammonification of soluble organic Nitrogen | 0 | 0 | 0 |
| 7 | Hydrolysis of entrapped organics | 0 | 0 | 0 |
| 8 | Hydrolysis of entrapped organic nitrogen | 0 | 0 | 0 |

Maximum absolute residual: `5.5511151231257827e-17`; acceptance tolerance: `1.0e-15`.

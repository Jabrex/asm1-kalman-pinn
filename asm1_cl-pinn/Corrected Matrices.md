---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# Corrected Matrices

`∅ (0)` means the workbook cell is blank while its effective coefficient is zero. A literal `0` remains distinguishable.

## Corrected Matrix: Original Notation

| Process / conserved quantity | SI | SS | XI | XS | XB,H | XB,A | XP | SO | SNO | SNH | SND | XND | SALK | SN2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aerobic growth of heterotrophs | ∅ (0) | -1/YH | ∅ (0) | ∅ (0) | 1 | ∅ (0) | ∅ (0) | -(1-YH)/YH | ∅ (0) | -iXB | ∅ (0) | ∅ (0) | -iXB*iCharge_SNHx | ∅ (0) |
| Anoxic growth of heterotrophs | ∅ (0) | -1/YH | ∅ (0) | ∅ (0) | 1 | ∅ (0) | ∅ (0) | ∅ (0) | -(1-YH)/(iNO3,N2*YH) | -iXB | ∅ (0) | ∅ (0) | -(1-YH)/(iNO3,N2*YH)*iCharge_SNOx-iXB*iCharge_SNHx | (1-YH)/(iNO3,N2*YH) |
| Aerobic growth of autotrophs | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | ∅ (0) | -(-iCOD_NO3-YA)/YA | 1/YA | -iXB-1/YA | ∅ (0) | ∅ (0) | -(iXB+1/YA)*iCharge_SNHx+(1/YA)*iCharge_SNOx | ∅ (0) |
| Decay of heterotrophs | ∅ (0) | ∅ (0) | ∅ (0) | 1-fP | -1 | ∅ (0) | fP | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | iXB-fP*iXP | ∅ (0) | ∅ (0) |
| Decay of autotrophs | ∅ (0) | ∅ (0) | ∅ (0) | 1-fP | ∅ (0) | -1 | fP | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | iXB-fP*iXP | ∅ (0) | ∅ (0) |
| Ammonification of soluble organic Nitrogen | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | -1 | ∅ (0) | iCharge_SNHx | ∅ (0) |
| Hydrolysis of entrapped organics | ∅ (0) | 1 | ∅ (0) | -1 | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) |
| Hydrolysis of entrapped organic nitrogen | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | -1 | ∅ (0) | ∅ (0) |

## Corrected Matrix: Standardised Notation

| Process / conserved quantity | SU | SB | XU,Inf | XCB | XOHO | XANO | XU,E | SO2 | SNOX | SNHX | SB,N | XCB,N | SAlk | SN2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aerobic growth of heterotrophs | ∅ (0) | -1/YOHO | ∅ (0) | ∅ (0) | 1 | ∅ (0) | ∅ (0) | -(1-YOHO)/YOHO | ∅ (0) | -iN_XBio | ∅ (0) | ∅ (0) | -iN_XBio*iCharge_SNHx | ∅ (0) |
| Anoxic growth of heterotrophs | ∅ (0) | -1/YOHO | ∅ (0) | ∅ (0) | 1 | ∅ (0) | ∅ (0) | ∅ (0) | -(1-YOHO)/(iNO3,N2*YOHO) | -iN_XBio | ∅ (0) | ∅ (0) | -(1-YOHO)/(iNO3,N2*YOHO)*iCharge_SNOx-iN_XBio*iCharge_SNHx | (1-YOHO)/(iNO3,N2*YOHO) |
| Aerobic growth of autotrophs | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | ∅ (0) | -(-iCOD_NO3-YANO)/YANO | 1/YANO | -iN_XBio-1/YANO | ∅ (0) | ∅ (0) | -(iN_XBio+1/YANO)*iCharge_SNHx+(1/YANO)*iCharge_SNOx | ∅ (0) |
| Decay of heterotrophs | ∅ (0) | ∅ (0) | ∅ (0) | 1-fXU_Bio,lys | -1 | ∅ (0) | fXU_Bio,lys | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | iN_XBio-fXU_Bio,lys*iN_XUE | ∅ (0) | ∅ (0) |
| Decay of autotrophs | ∅ (0) | ∅ (0) | ∅ (0) | 1-fXU_Bio,lys | ∅ (0) | -1 | fXU_Bio,lys | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | iN_XBio-fXU_Bio,lys*iN_XUE | ∅ (0) | ∅ (0) |
| Ammonification of soluble organic Nitrogen | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | -1 | ∅ (0) | iCharge_SNHx | ∅ (0) |
| Hydrolysis of entrapped organics | ∅ (0) | 1 | ∅ (0) | -1 | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) |
| Hydrolysis of entrapped organic nitrogen | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | ∅ (0) | 1 | -1 | ∅ (0) | ∅ (0) |


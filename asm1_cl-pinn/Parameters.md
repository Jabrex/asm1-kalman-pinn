---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# Parameters

## Stoichiometric parameters

| Original       | Standardised   | Code ID        | Description                               | Unit            | Full value            | Source formula | Source     |
| -------------- | -------------- | -------------- | ----------------------------------------- | --------------- | --------------------- | -------------- | ---------- |
| `YH`           | `YOHO `        | `YH`           | Yield for XOHO growth                     | g XOHO.g XCB-1  | 0.67000000000000004   | —              | Sheet1!G25 |
| `fP`           | `fXU_Bio,lys`  | `fP`           | Fraction of XU generated in biomass decay | g XU.g XBio-1   | 0.080000000000000002  | —              | Sheet1!G26 |
| `YA`           | `YANO `        | `YA`           | Yield of XANO growth per SNO3             | g XAUT.g SNO3-1 | 0.23999999999999999   | —              | Sheet1!G27 |
| `iXB`          | `iN_XBio`      | `iXB`          | N content of biomass (XOHO, XPAO, XANO)   | g N.g XBio-1    | 0.085999999999999993  | —              | Sheet1!G28 |
| `iXP`          | `iN_XUE`       | `iXP`          | N content of products from biomass        | g N.g XUE-1     | 0.059999999999999998  | —              | Sheet1!G29 |
| `iNO3,N2`      | `iNO3,N2`      | `iNO3_N2`      | Conversion factor for NO3 reduction to N2 | g COD.g N-1     | 2.8571428571428572    | `=40/14`       | Sheet1!G30 |
| `iCOD_NO3`     | `iCOD_NO3`     | `iCOD_NO3`     | Conversion factor for NO3 in COD          | g COD.g N-1     | -4.5714285714285712   | `=-64/14`      | Sheet1!G31 |
| `iCOD_N2`      | `iCOD_N2`      | `iCOD_N2`      | Conversion factor for N2 in COD           | g COD.g N-1     | -1.7142857142857142   | `=-24/14`      | Sheet1!G32 |
| `iCharge_SNHx` | `iCharge_SNHx` | `iCharge_SNHx` | Conversion factor for NHx in charge       | Charge.g N-1    | 0.071428571428571425  | `=1/14`        | Sheet1!G33 |
| `iCharge_SNOx` | `iCharge_SNOx` | `iCharge_SNOx` | Conversion factor for NO3 in charge       | Charge.g N-1    | -0.071428571428571425 | `=-1/14`       | Sheet1!G34 |

## Kinetic parameters

| Original | Standardised | Code ID | Description | Unit | Full value | Source formula | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `kh` | `qXCB_SB,hyd` | `kh` | Maximum specific hydrolysis rate | g XCB.g XOHO-1.d-1 | 3 | — | Sheet1!G35 |
| `KX` | `KXCB,hyd` | `KX` | Saturation coefficient for XB/XOHO | g XCB.g XOHO-1 | 0.029999999999999999 | — | Sheet1!G36 |
| `ηh` | `nqhyd,Ax` | `etah` | Correction factor for hydrolysis under anoxic conditions | - | 0.40000000000000002 | — | Sheet1!G37 |
| `μH` | `μOHO,Max` | `muH` | Maximum growth rate of XOHO | d-1 | 6 | — | Sheet1!G38 |
| `ηg` | `nμOHO,Ax` | `etag` | Reduction factor for anoxic growth of XOHO | - | 0.80000000000000004 | — | Sheet1!G39 |
| `Ks` | `KSB,OHO` | `Ks` | Half-saturation coefficient for SB | g SB.m-3 | 20 | — | Sheet1!G40 |
| `bH` | `bOHO` | `bH` | Decay rate for XOHO | d-1 | 0.62 | — | Sheet1!G41 |
| `KO,H` | `KO2,OHO` | `KO_H` | Half-saturation coefficient for SO2 | g SO2.m-3 | 0.20000000000000001 | — | Sheet1!G42 |
| `KNO` | `KNOx,OHO` | `KNO` | Half-saturation coefficient for SNOx | g SNOx.m-3 | 0.5 | — | Sheet1!G43 |
| `KNH,H` | `KNHx,OHO` | `KNH_H` | Half-saturation coefficient  for NH4* | g SNHx.m-3 | 0.050000000000000003 | — | Sheet1!G44 |
| `μA` | `μANO,Max` | `muA` | Maximum growth rate of XANO | d-1 | 0.80000000000000004 | — | Sheet1!G45 |
| `bA` | `bANO` | `bA` | Decay rate for XANO | d-1 | 0.14999999999999999 | — | Sheet1!G46 |
| `ka` | `qam` | `ka` | Rate constant for ammonification | m3.g XCB,N-1.d-1 | 0.080000000000000002 | — | Sheet1!G47 |
| `KO,A` | `KO2,ANO` | `KO_A` | Half-saturation coefficient for SO2 | g SO2.m-3 | 0.40000000000000002 | — | Sheet1!G48 |
| `KNH` | `KNHx,ANO` | `KNH` | Half-saturation coefficient for SNHx | g SNHx.m-3 | 1 | — | Sheet1!G49 |


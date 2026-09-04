---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# Processes and Rates

Rate cells are source text, not Excel formulas. Each expression is preserved character-for-character.

## 1. Aerobic growth of heterotrophs

### Original source expression — `Sheet1!Z12`

```text
μH*[SS/(KS+SS)]*[SO/(KO,H+SO)]*[SNH/(KNH,H+SNH)]*XB,H
```

Code-safe expression:

```text
muH*(S_S/(Ks+S_S))*(S_O/(KO_H+S_O))*(S_NH/(KNH_H+S_NH))*X_B_H
```

### Standardised source expression — `Sheet1!Z29`

```text
μOHO,Max*[SB/(KSB,OHO+SB)]*[SO2/(KO2,OHO+SO2)]*[SNHX/(KNHx,OHO+SNHX)]*XOHO
```

Code-safe expression:

```text
muH*(S_S/(Ks+S_S))*(S_O/(KO_H+S_O))*(S_NH/(KNH_H+S_NH))*X_B_H
```

## 2. Anoxic growth of heterotrophs

### Original source expression — `Sheet1!Z13`

```text
μH*[SS/(KS+SS)]*[KO,H/(KO,H+SO)]*[SNO/(KNO+SNO)]*[SNH/(KNH,H+SNH)]*ηg*XB,H
```

Code-safe expression:

```text
muH*(S_S/(Ks+S_S))*(KO_H/(KO_H+S_O))*(S_NO/(KNO+S_NO))*(S_NH/(KNH_H+S_NH))*etag*X_B_H
```

### Standardised source expression — `Sheet1!Z30`

```text
μOHO,Max*[SB/(KSB,OHO+SB)]*[KO2,OHO/(KO2,OHO+SO2)]*[SNOx/(KNOx,OHO+SNOx)]*[SNHX/(KNHx,OHO+SNHX)]*nμOHO,Ax*XOHO
```

Code-safe expression:

```text
muH*(S_S/(Ks+S_S))*(KO_H/(KO_H+S_O))*(S_NO/(KNO+S_NO))*(S_NH/(KNH_H+S_NH))*etag*X_B_H
```

## 3. Aerobic growth of autotrophs

### Original source expression — `Sheet1!Z14`

```text
μA*[SNH/(KNH+SNH)]*[SO/(KO,A+SO)]*XB,A
```

Code-safe expression:

```text
muA*(S_NH/(KNH+S_NH))*(S_O/(KO_A+S_O))*X_B_A
```

### Standardised source expression — `Sheet1!Z31`

```text
μANO,Max*[SNHX/(KNHx,ANO+SNHX)]*[SO2/(KO2,ANO+SO2)]*XANO
```

Code-safe expression:

```text
muA*(S_NH/(KNH+S_NH))*(S_O/(KO_A+S_O))*X_B_A
```

## 4. Decay of heterotrophs

### Original source expression — `Sheet1!Z15`

```text
bH*XB,H
```

Code-safe expression:

```text
bH*X_B_H
```

### Standardised source expression — `Sheet1!Z32`

```text
bOHO*XOHO
```

Code-safe expression:

```text
bH*X_B_H
```

## 5. Decay of autotrophs

### Original source expression — `Sheet1!Z16`

```text
bA*XB,A
```

Code-safe expression:

```text
bA*X_B_A
```

### Standardised source expression — `Sheet1!Z33`

```text
bANO*XANO
```

Code-safe expression:

```text
bA*X_B_A
```

## 6. Ammonification of soluble organic Nitrogen

### Original source expression — `Sheet1!Z17`

```text
ka*SND*XB,H
```

Code-safe expression:

```text
ka*S_ND*X_B_H
```

### Standardised source expression — `Sheet1!Z34`

```text
qam*SB,N*XOHO
```

Code-safe expression:

```text
ka*S_ND*X_B_H
```

## 7. Hydrolysis of entrapped organics

### Original source expression — `Sheet1!Z18`

```text
kH*[(XS/XB,H)/(KX+XS/XB,H)]*([SO/(KO,H+SO)]+ηh*[KO,H/(KO,H+SO)]*[SNO/(KNO+SNO)])*XB,H
```

Code-safe expression:

```text
kh*((X_S/X_B_H)/(KX+X_S/X_B_H))*((S_O/(KO_H+S_O))+etah*(KO_H/(KO_H+S_O))*(S_NO/(KNO+S_NO)))*X_B_H
```

### Standardised source expression — `Sheet1!Z35`

```text
qXCB_SB,hyd*[(XCB/XOHO)/(KXCB,hyd+XCB/XOHO)]*([SO2/(KO2,OHO+SO2)]+nqhyd,Ax*[KO2,OHO/(KO2,OHO+SO2)]*[SNOx/(KNOx,OHO+SNOx)])*XOHO
```

Code-safe expression:

```text
kh*((X_S/X_B_H)/(KX+X_S/X_B_H))*((S_O/(KO_H+S_O))+etah*(KO_H/(KO_H+S_O))*(S_NO/(KNO+S_NO)))*X_B_H
```

## 8. Hydrolysis of entrapped organic nitrogen

### Original source expression — `Sheet1!Z19`

```text
kH*(XND/XS)*[(XS/XB,H)/(KX+XS/XB,H)]*([SO/(KO,H+SO)]+ηh*[KO,H/(KO,H+SO)]*[SNO/(KNO+SNO)])*XB,H
```

Code-safe expression:

```text
kh*(X_ND/X_S)*((X_S/X_B_H)/(KX+X_S/X_B_H))*((S_O/(KO_H+S_O))+etah*(KO_H/(KO_H+S_O))*(S_NO/(KNO+S_NO)))*X_B_H
```

### Standardised source expression — `Sheet1!Z36`

```text
qXCB_SB,hyd*(XCB,N/XCB)*[(XCB/XOHO)/(KXCB,hyd+XCB/XOHO)]*([SO2/(KO2,OHO+SO2)]+nqhyd,Ax*[KO2,OHO/(KO2,OHO+SO2)]*[SNOx/(KNOx,OHO+SNOx)])*XOHO
```

Code-safe expression:

```text
kh*(X_ND/X_S)*((X_S/X_B_H)/(KX+X_S/X_B_H))*((S_O/(KO_H+S_O))+etah*(KO_H/(KO_H+S_O))*(S_NO/(KNO+S_NO)))*X_B_H
```


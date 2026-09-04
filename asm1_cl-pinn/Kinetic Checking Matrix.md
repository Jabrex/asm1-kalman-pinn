---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# Kinetic Checking Matrix

| Category | Source color | Meaning | Rate consequence |
| --- | --- | --- | --- |
| consumed | #00FF00 | Consumed component (every state variable with a negative sign) | Limitation monod function |
| biomass | #00CCFF | Biomass involved in the process | proportional to the biomass concentration |
| other_required | #FFFF00 | Other required component | Limitation monod function |
| inhibitory | #FF0000 | Inhibitory component | Inhibitory monod function |

| Process | S_I | S_S | X_I | X_S | X_B.H | X_B.A | X_P | S_O | S_NO | S_NH | S_ND | X_ND | S_ALK | S_N2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Aerobic growth of heterotrophs | — | consumed | — | — | biomass | — | — | consumed | — | consumed | — | — | consumed* | — |
| Anoxic growth of heterotrophs | — | consumed | — | — | biomass | — | — | inhibitory | consumed | consumed | — | — | — | — |
| Aerobic growth of autotrophs | — | — | — | — | — | biomass | — | consumed | — | consumed | — | — | consumed* | — |
| Decay of heterotrophs | — | — | — | — | biomass | — | — | — | — | — | — | — | — | — |
| Decay of autotrophs | — | — | — | — | — | biomass | — | — | — | — | — | — | — | — |
| Ammonification of soluble organic Nitrogen | — | — | — | — | biomass | — | — | — | — | — | consumed | — | — | — |
| Hydrolysis of entrapped organics | — | — | — | consumed | biomass | — | — | other_required | other_required | — | — | — | — | — |
| Hydrolysis of entrapped organic nitrogen | — | — | — | — | biomass | — | — | other_required | other_required | — | — | consumed | — | — |

`X82` and `X84` are green consumed-component cells marked `*`. The workbook states that these missing alkalinity kinetic terms have not been corrected; no term is invented here.

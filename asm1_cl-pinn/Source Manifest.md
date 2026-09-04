---
model: ASM1
schema_version: "1.0.0"
source_sha256: dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a
canonical_data: data/asm1.json
---

# Source Manifest

| Property | Value |
| --- | --- |
| Source path | C:\Users\musta\Desktop\asm1_cl_pinn\asm1.xlsx |
| SHA-256 | dff2424c5fa1ed83846ebac7269ac3284317dc8799f18d7edaabb18d60ba892a |
| Size | 55914 |
| Worksheet | Sheet1 |
| Used range | A1:AC91 |
| Formula cells | 56 |
| Calculation-chain cells | 56 |
| Named ranges | 10 |
| Formula errors | 0 |

## Named ranges

| Name | Formula | Value | Source |
| --- | --- | --- | --- |
| ASM1_f_P | =Sheet1!$G$26 | 0.080000000000000002 | Sheet1!G26 |
| ASM1_i_Charge_NHx | =Sheet1!$G$33 | 0.071428571428571425 | Sheet1!G33 |
| ASM1_i_Charge_NOx | =Sheet1!$G$34 | -0.071428571428571425 | Sheet1!G34 |
| ASM1_i_COD_N2 | =Sheet1!$G$32 | -1.7142857142857142 | Sheet1!G32 |
| ASM1_i_COD_NOx | =Sheet1!$G$31 | -4.5714285714285712 | Sheet1!G31 |
| ASM1_i_NOx.N2 | =Sheet1!$G$30 | 2.8571428571428572 | Sheet1!G30 |
| ASM1_i_XB | =Sheet1!$G$28 | 0.085999999999999993 | Sheet1!G28 |
| ASM1_i_XP | =Sheet1!$G$29 | 0.059999999999999998 | Sheet1!G29 |
| ASM1_Y_A | =Sheet1!$G$27 | 0.23999999999999999 | Sheet1!G27 |
| ASM1_Y_H | =Sheet1!$G$25 | 0.67000000000000004 | Sheet1!G25 |

## Formula cells

| Cell | Kind | Formula | Value |
| --- | --- | --- | --- |
| Sheet1!G30 | normal | `=40/14` | 2.8571428571428572 |
| Sheet1!G31 | normal | `=-64/14` | -4.5714285714285712 |
| Sheet1!G32 | normal | `=-24/14` | -1.7142857142857142 |
| Sheet1!G33 | normal | `=1/14` | 0.071428571428571425 |
| Sheet1!G34 | normal | `=-1/14` | -0.071428571428571425 |
| Sheet1!AB52 | normal | `=ASM1_i_XB` | 0.085999999999999993 |
| Sheet1!AB53 | normal | `=ASM1_i_XB` | 0.085999999999999993 |
| Sheet1!AB54 | normal | `=ASM1_i_XP` | 0.059999999999999998 |
| Sheet1!AA56 | normal | `=ASM1_i_COD_NOx` | -4.5714285714285712 |
| Sheet1!AC56 | normal | `=ASM1_i_Charge_NOx` | -0.071428571428571425 |
| Sheet1!AC57 | normal | `=ASM1_i_Charge_NHx` | 0.071428571428571425 |
| Sheet1!AA61 | normal | `=ASM1_i_COD_N2` | -1.7142857142857142 |
| Sheet1!M63 | normal | `=-1/ASM1_Y_H` | -1.4925373134328357 |
| Sheet1!S63 | normal | `=-(1-ASM1_Y_H)/ASM1_Y_H` | -0.49253731343283574 |
| Sheet1!U63 | normal | `=-ASM1_i_XB` | -0.085999999999999993 |
| Sheet1!X63 | normal | `=-ASM1_i_XB*ASM1_i_Charge_NHx` | -0.0061428571428571417 |
| Sheet1!AA63 | shared_master | `=L63*$AA$47+M63*$AA$48+N63*$AA$49+O63*$AA$50+P63*$AA$52+Q63*$AA$53+R63*$AA$54+S63*$AA$55+T63*$AA$56+U63*$AA$57+V63*$AA$58+W63*$AA$59+X63*$AA$60+Y63*$AA$61` | 5.5511151231257827e-17 |
| Sheet1!AB63 | shared_master | `=L63*$AB$47+M63*$AB$48+N63*$AB$49+O63*$AB$50+P63*$AB$52+Q63*$AB$53+R63*$AB$54+S63*$AB$55+T63*$AB$56+U63*$AB$57+V63*$AB$58+W63*$AB$59+X63*$AB$60+Y63*$AB$61` | 0 |
| Sheet1!AC63 | shared_master | `=T63*$AC$56+U63*$AC$57+X63*$AC$60` | 0 |
| Sheet1!M64 | normal | `=-1/ASM1_Y_H` | -1.4925373134328357 |
| Sheet1!T64 | normal | `=-(1-ASM1_Y_H)/(ASM1_i_NOx.N2*ASM1_Y_H)` | -0.17238805970149251 |
| Sheet1!U64 | normal | `=-ASM1_i_XB` | -0.085999999999999993 |
| Sheet1!X64 | normal | `=-(1-ASM1_Y_H)/(ASM1_i_NOx.N2*ASM1_Y_H)*ASM1_i_Charge_NOx-ASM1_i_XB*ASM1_i_Charge_NHx` | 0.0061705756929637508 |
| Sheet1!Y64 | normal | `=(1-ASM1_Y_H)/(ASM1_i_NOx.N2*ASM1_Y_H)` | 0.17238805970149251 |
| Sheet1!AA64 | shared_follower | `=L64*$AA$47+M64*$AA$48+N64*$AA$49+O64*$AA$50+P64*$AA$52+Q64*$AA$53+R64*$AA$54+S64*$AA$55+T64*$AA$56+U64*$AA$57+V64*$AA$58+W64*$AA$59+X64*$AA$60+Y64*$AA$61` | 0 |
| Sheet1!AB64 | shared_follower | `=L64*$AB$47+M64*$AB$48+N64*$AB$49+O64*$AB$50+P64*$AB$52+Q64*$AB$53+R64*$AB$54+S64*$AB$55+T64*$AB$56+U64*$AB$57+V64*$AB$58+W64*$AB$59+X64*$AB$60+Y64*$AB$61` | 0 |
| Sheet1!AC64 | shared_follower | `=T64*$AC$56+U64*$AC$57+X64*$AC$60` | 0 |
| Sheet1!S65 | normal | `=-(-ASM1_i_COD_NOx-ASM1_Y_A)/ASM1_Y_A` | -18.047619047619047 |
| Sheet1!T65 | normal | `=1/ASM1_Y_A` | 4.166666666666667 |
| Sheet1!U65 | normal | `=-ASM1_i_XB-1/ASM1_Y_A` | -4.2526666666666673 |
| Sheet1!X65 | normal | `=-(ASM1_i_XB+1/ASM1_Y_A)*ASM1_i_Charge_NHx+(1/ASM1_Y_A)*ASM1_i_Charge_NOx` | -0.60138095238095235 |
| Sheet1!AA65 | shared_follower | `=L65*$AA$47+M65*$AA$48+N65*$AA$49+O65*$AA$50+P65*$AA$52+Q65*$AA$53+R65*$AA$54+S65*$AA$55+T65*$AA$56+U65*$AA$57+V65*$AA$58+W65*$AA$59+X65*$AA$60+Y65*$AA$61` | 0 |
| Sheet1!AB65 | shared_follower | `=L65*$AB$47+M65*$AB$48+N65*$AB$49+O65*$AB$50+P65*$AB$52+Q65*$AB$53+R65*$AB$54+S65*$AB$55+T65*$AB$56+U65*$AB$57+V65*$AB$58+W65*$AB$59+X65*$AB$60+Y65*$AB$61` | 0 |
| Sheet1!AC65 | shared_follower | `=T65*$AC$56+U65*$AC$57+X65*$AC$60` | 0 |
| Sheet1!O66 | normal | `=1-ASM1_f_P` | 0.92000000000000004 |
| Sheet1!R66 | normal | `=ASM1_f_P` | 0.080000000000000002 |
| Sheet1!W66 | normal | `=ASM1_i_XB-ASM1_f_P*ASM1_i_XP` | 0.081199999999999994 |
| Sheet1!AA66 | shared_follower | `=L66*$AA$47+M66*$AA$48+N66*$AA$49+O66*$AA$50+P66*$AA$52+Q66*$AA$53+R66*$AA$54+S66*$AA$55+T66*$AA$56+U66*$AA$57+V66*$AA$58+W66*$AA$59+X66*$AA$60+Y66*$AA$61` | 4.163336342344337e-17 |
| Sheet1!AB66 | shared_follower | `=L66*$AB$47+M66*$AB$48+N66*$AB$49+O66*$AB$50+P66*$AB$52+Q66*$AB$53+R66*$AB$54+S66*$AB$55+T66*$AB$56+U66*$AB$57+V66*$AB$58+W66*$AB$59+X66*$AB$60+Y66*$AB$61` | 0 |
| Sheet1!AC66 | shared_follower | `=T66*$AC$56+U66*$AC$57+X66*$AC$60` | 0 |
| Sheet1!O67 | normal | `=1-ASM1_f_P` | 0.92000000000000004 |
| Sheet1!R67 | normal | `=ASM1_f_P` | 0.080000000000000002 |
| Sheet1!W67 | normal | `=ASM1_i_XB-ASM1_f_P*ASM1_i_XP` | 0.081199999999999994 |
| Sheet1!AA67 | shared_follower | `=L67*$AA$47+M67*$AA$48+N67*$AA$49+O67*$AA$50+P67*$AA$52+Q67*$AA$53+R67*$AA$54+S67*$AA$55+T67*$AA$56+U67*$AA$57+V67*$AA$58+W67*$AA$59+X67*$AA$60+Y67*$AA$61` | 4.163336342344337e-17 |
| Sheet1!AB67 | shared_follower | `=L67*$AB$47+M67*$AB$48+N67*$AB$49+O67*$AB$50+P67*$AB$52+Q67*$AB$53+R67*$AB$54+S67*$AB$55+T67*$AB$56+U67*$AB$57+V67*$AB$58+W67*$AB$59+X67*$AB$60+Y67*$AB$61` | 0 |
| Sheet1!AC67 | shared_follower | `=T67*$AC$56+U67*$AC$57+X67*$AC$60` | 0 |
| Sheet1!X68 | normal | `=ASM1_i_Charge_NHx` | 0.071428571428571425 |
| Sheet1!AA68 | shared_follower | `=L68*$AA$47+M68*$AA$48+N68*$AA$49+O68*$AA$50+P68*$AA$52+Q68*$AA$53+R68*$AA$54+S68*$AA$55+T68*$AA$56+U68*$AA$57+V68*$AA$58+W68*$AA$59+X68*$AA$60+Y68*$AA$61` | 0 |
| Sheet1!AB68 | shared_follower | `=L68*$AB$47+M68*$AB$48+N68*$AB$49+O68*$AB$50+P68*$AB$52+Q68*$AB$53+R68*$AB$54+S68*$AB$55+T68*$AB$56+U68*$AB$57+V68*$AB$58+W68*$AB$59+X68*$AB$60+Y68*$AB$61` | 0 |
| Sheet1!AC68 | shared_follower | `=T68*$AC$56+U68*$AC$57+X68*$AC$60` | 0 |
| Sheet1!AA69 | shared_follower | `=L69*$AA$47+M69*$AA$48+N69*$AA$49+O69*$AA$50+P69*$AA$52+Q69*$AA$53+R69*$AA$54+S69*$AA$55+T69*$AA$56+U69*$AA$57+V69*$AA$58+W69*$AA$59+X69*$AA$60+Y69*$AA$61` | 0 |
| Sheet1!AB69 | shared_follower | `=L69*$AB$47+M69*$AB$48+N69*$AB$49+O69*$AB$50+P69*$AB$52+Q69*$AB$53+R69*$AB$54+S69*$AB$55+T69*$AB$56+U69*$AB$57+V69*$AB$58+W69*$AB$59+X69*$AB$60+Y69*$AB$61` | 0 |
| Sheet1!AC69 | shared_follower | `=T69*$AC$56+U69*$AC$57+X69*$AC$60` | 0 |
| Sheet1!AA70 | shared_follower | `=L70*$AA$47+M70*$AA$48+N70*$AA$49+O70*$AA$50+P70*$AA$52+Q70*$AA$53+R70*$AA$54+S70*$AA$55+T70*$AA$56+U70*$AA$57+V70*$AA$58+W70*$AA$59+X70*$AA$60+Y70*$AA$61` | 0 |
| Sheet1!AB70 | shared_follower | `=L70*$AB$47+M70*$AB$48+N70*$AB$49+O70*$AB$50+P70*$AB$52+Q70*$AB$53+R70*$AB$54+S70*$AB$55+T70*$AB$56+U70*$AB$57+V70*$AB$58+W70*$AB$59+X70*$AB$60+Y70*$AB$61` | 0 |
| Sheet1!AC70 | shared_follower | `=T70*$AC$56+U70*$AC$57+X70*$AC$60` | 0 |

## Source anomalies preserved

| ID | Source text | Cells | Handling |
| --- | --- | --- | --- |
| pocess_rate_typo | Pocess rate | Sheet1!Z11, Sheet1!Z28 | preserved; canonical concept is process_rate |
| kh_case | kh in parameter table; kH in process rates | Sheet1!D35, Sheet1!Z18, Sheet1!Z19 | preserved; all aliases map to k_h |
| ks_case | Ks in parameter table; KS in process rates | Sheet1!D40, Sheet1!Z12, Sheet1!Z13 | preserved; all aliases map to K_S |
| matrix_only_sn2 | SN2 occurs in matrices but has no state-variable table row | Sheet1!Y11, Sheet1!Y28 | preserved as operational component 14 with added_term_not_listed status |
| sno_case_variants | SNOx / SNOX / S_NO | Sheet1!E19, Sheet1!T28, Sheet1!T62 | preserved as aliases of S_NO |
| snh_case_variants | SNHx / SNHX / S_NH | Sheet1!E18, Sheet1!U28, Sheet1!U62 | preserved as aliases of S_NH |
| named_range_aliases | iNO3,N2 / ASM1_i_NOx.N2 and NO3 / NOx variants | Sheet1!D30, Sheet1!G30 | source names retained; code identifiers are separate |
| oxygen_unit_prefix | - g COD.m-3 | Sheet1!F14 | unit text preserved verbatim |
| double_negative_process3 | -(-iCOD_NO3-YA)/YA | Sheet1!S14 | expression preserved and independently evaluated |
| text_rate_cells | All 16 rate expressions are text, not Excel formulas | Sheet1!Z12:Z19, Sheet1!Z29:Z36 | source_kind is text_expression |
| comma_and_dot_identifiers | Commas and dots are part of symbols and named ranges | Sheet1!D22, Sheet1!D30, Sheet1!G30 | longest-match alias table; never split as CSV |
| missing_alkalinity_kinetics | * Missing kinetic terms that have not been corrected | Sheet1!X82, Sheet1!X84, Sheet1!K91:Y91 | flagged; no kinetic term invented |
| knh_h_special_value | *same KNH,H value as ASM2d has been chosen | Sheet1!D44:G44, Sheet1!C51 | source note and value preserved |
| standardised_parameter_trailing_spaces | YOHO␠ and YANO␠ contain a trailing U+0020 space | Sheet1!E25, Sheet1!E27 | raw standardised_notation preserves the final space; aliases and code identifiers are trimmed separately |

## Legacy artifacts

| ID | Program | Archive path | Size | SHA-256 | Anchor | Role |
| --- | --- | --- | --- | --- | --- | --- |
| legacy_equation_1 | Equation.3 | xl/embeddings/oleObject1.bin | 3072 | 613d5d92c240347bfa0526d9197199b2cd55f2a1496825a67e2d960f59ab302c | Y73 | none; hidden zero-size legacy OLE object |
| legacy_equation_2 | Equation.3 | xl/embeddings/oleObject2.bin | 3072 | f3afceab0fa991abe1fc72daecfb8515a993b0283ba96358af528ead9ace3929 | Y73 | none; hidden zero-size legacy OLE object |
| legacy_equation_3 | Equation.3 | xl/embeddings/oleObject3.bin | 3072 | 613d5d92c240347bfa0526d9197199b2cd55f2a1496825a67e2d960f59ab302c | Y73 | none; hidden zero-size legacy OLE object |

## Literature cross-check

- Hauduc et al. (2010): https://pubmed.ncbi.nlm.nih.gov/20182061/
- Corominas et al. (2010): https://publications.polymtl.ca/18466/

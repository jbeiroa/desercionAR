
# Variable dictionary

## Household Features (Base de Hogares)

- **Identification:** `CODUSU` (Vessel/Dwelling ID), `NRO_HOGAR` (Household ID), `PONDERA` (Statistical weight/Survey weight), `AGLOMERADO` (Urban area code).
- **Housing Quality:** `IV1` (Housing type), `IV2` (Number of rooms), `IV3` (Floor material), `IV4` (Roof material), `IV6-IV11` (Water/Sanitation infrastructure), `IV12_1-3` (Environmental risk: proximity to trash heaps, flooding, or slums).
- **Living Conditions:** `II1-II2` (Exclusive use rooms and bedrooms), `II7` (Tenure status), `II8` (Cooking fuel).
- **Income Sources (Last 3 Months):** `V1-V14` (Binary flags for income from: employment, pensions, social subsidies, food aid, savings, or loans).
- **Demographics:** `IX_TOT` (Total household members), `IX_MEN10` (Children under 10), `DECCFR` (Income decile).

## Individual Features (Base de Individuos)

- **Demographics:** `CH03` (Relation to head of household), `CH04` (Sex), `CH06` (Age), `CH07` (Civil status).
- **Education:** `CH09` (Literacy), `CH10` (School attendance), `CH11` (Public/Private sector), `NIVEL_ED` (Educational level attained).
- **Labor:** `ESTADO` (Activity status), `PP02E` (Reason for not seeking work), `PP07H/I` (Pension contributions).

## Engineered Features (Derived)

- **Head of Household (jefx) context:** `CH06_jefx` (Age), `ESTADO_jefx` (Labor status), `NIVEL_ED_jefx` (Education level), `JEFA_MUJER` (Is the head of household female?).
- **Household Composition:** `HOGAR_MONOP` (Single-parent household), `ratio_ocupados` (Employment ratio within household).
- **Unsatisfied Basic Needs (NBI):** `NBI_` flags covering Housing, Sanitation, Tenure, and Labor precariousness.
- **Target:** DESERTO (Dropout indicator).
# Model variables and manuscript quantities

The manuscript and figure labels use physical descriptions. Native model fields and analytical table columns retain their original names so that the published source data and code remain directly usable. This crosswalk links those technical names to the scientific quantities; it does not redefine the calculations or rename data columns. The [source-data index](source_data_index.md) maps figures and Supplementary Tables to the relevant records.

## Land–atmosphere carbon accounting

| Internal field or expression | Reader-facing quantity | Definition and sign | Units |
|---|---|---|---|
| `D_Fland` | Dynamic biome-reference land uptake used in atmospheric bookkeeping | The separately evolved, unit-area biome-reference uptake anomaly, weighted by actual biome areas. Positive values enter the atmospheric budget as uptake. This is not an independently measured forest sink or a second actual land inventory. | PgC yr⁻¹ |
| `D_Eluc` | Land-use bookkeeping term in the atmospheric carbon budget | The reference uptake minus actual net biome production relative to the preindustrial baseline. Positive values enter the atmospheric budget as a source. In this implementation it is a residual connecting the reference and actual land systems; it must not be read as deforestation emissions alone. | PgC yr⁻¹ |
| `rf_NBP` | Actual complete-land net biome production | Actual net uptake summed across native land classes, with living carbon, debris, soil and products represented and permafrost excluded. | PgC yr⁻¹ |
| `rf_NBP_pi` | Preindustrial net-biome-production baseline | The common preindustrial reference used to express land exchange as an anomaly. | PgC yr⁻¹ |
| `D_Fland - D_Eluc` | Net land uptake, relative to the preindustrial baseline | Exactly `rf_NBP - rf_NBP_pi`. Positive values denote uptake from the atmosphere into the tracked land system relative to that baseline. The common baseline cancels in matched on-minus-off differences. | PgC yr⁻¹ |
| Annual integral of `D_Fland - D_Eluc` | Cumulative net land uptake | Sum of saved annual mean rates multiplied by their one-year intervals. The future integral uses the 77 intervals labelled 2023–2099, excluding the common 2022 climate anchor. | PgC |

The actual mapping is implemented in `oscar/_core/_model/_modules/regional_forest_age_native.py`. The dynamic reference and its weighting are in `oscar/_core/_model/_modules/regional_forest_age_reference.py`. Atmospheric budget signs and the separately represented permafrost term are defined in `oscar/_core/_model/_modules/OSCAR_CO2.py`.

Native land exchanges retain their regional coordinates; global paper quantities sum those regions. Local one-hectare experiments instead use their stated area normalization. Net land uptake is distinct from NPP, forest-only net exchange, a living-biomass stock change and an age-specific allocation of mixed soil or product carbon. The direct production adjustment in Fig. 3 is a separate diagnostic evaluated along the age-on state and environment.

## Atmospheric quantities and member identities

| Internal field | Reader-facing quantity | Conversion or support | Units |
|---|---|---|---|
| `D_CO2` | Atmospheric CO₂ concentration anomaly | Simulated change from the corresponding member's preindustrial CO₂ concentration. | ppm |
| `CO2_pi` | Preindustrial atmospheric CO₂ concentration | Member-specific parameter. Absolute atmospheric CO₂ is calculated as `D_CO2 + CO2_pi` within each member before summarising. | ppm |
| `D_Tg` | Model global mean surface temperature | Native temperature state. Future manuscript warming subtracts each member's common H1 historical 1850–1900 mean. Extended Data Fig. 3 retains unrecentred H1-minus-H0 differences; subtracting the same reference from both arms would leave those differences unchanged. | K; differences may be displayed as mK |
| `source_config` | Original joint-parameter member identifier | Links the saved configuration to its original supplied identity and order; it is not an outcome or an independent replicate dimension. | Identifier |

CO₂ is defined in `oscar/_core/_model/_modules/OSCAR_CO2.py`; the preindustrial parameter is supplied with the original ensemble. Temperature is defined in `oscar/_core/_model/_modules/OSCAR_clim.py`. Exact member order is supplied by `inputs/ensemble/ordered_members.csv`.

## Supplementary establishment-response labels

The two following labels are analysis contrasts, not OSCAR state variables. The Supplementary Information uses descriptive names and the mathematical symbols shown below.

| Code or source-data label | Manuscript description | Exact comparison |
|---|---|---|
| `future_W_only` (the additional run labelled `new` in the analysis) | Additional treatment X: future woody response retained, future other-living-tissue production unmodulated | Existing-cohort responses remain on; post-2022 woody response is unchanged; post-2022 other-living-tissue multiplier is one. No stock is instantaneously filled. |
| `N1_W` and columns beginning `N1_W_` | Future woody-response effect, C_W | X − 10: future woody response introduced while future other-tissue production remains unmodulated. |
| `F_cond` and columns beginning `F_cond_` | Conditional other-living-tissue effect, C_F | 11 − X: future other-tissue response added with the future woody response already enabled. |
| `N1` | Conditional future-cohort response | 11 − 10 = C_W + C_F for the same outcome, member and calendar. |
| `p00` | Reference maximum year | Each member's treatment-00 window-maximum year. It need not be a resolved overshoot peak. |

The units of a contrast follow its outcome and suffix: temperature in `*_K` is K (multiply by 1,000 for mK); `*_PgC` is a cumulative carbon quantity; `*_ppm` is a concentration; `*_years` is an annual-interval count. For the same metric and eligibility mask, per-member contrasts and ensemble means telescope. Medians and percentiles are not required to add. The ordered comparisons are not independent tissue shares.

The defined mathematical quantities S_R(a), g_W(a), g_F(a), NPP, W, F, A, Q and the 00/10/01/11 treatment notation retain the meanings given in Methods. They are not unexpanded native variable names. In particular, F denotes all other living tissue, not leaves alone; Q is the projected above-ground secondary standing-harvest delivery metric, not harvested-wood-product production.

## Region identifiers

These identifiers are kept in model coordinates and source data. Supplementary Table 13 displays the region names.

| Internal coordinate | Model region |
|---|---|
| `ROWO` | Other |
| `AFRICA` | Africa |
| `CHINA+` | China+ |
| `EUROPE` | Europe |
| `INDIA+` | India+ |
| `LATIN_AM` | Latin America |
| `MIDDLE_EAST` | Middle East |
| `NORTH_AM` | North America |
| `PAC_OECD` | Pacific OECD |
| `REF_ECON` | Reforming Economies |
| `REST_ASIA` | Rest of Asia |

Country and territory assignments are in `oscar/_core/_regions/OSCAR_reg_dict.csv`. The ten IAMC regions plus Other are model spatial supports, not additional independent climate-ensemble members. Observation comparisons retain the coverage distinctions described in Extended Data Fig. 2.

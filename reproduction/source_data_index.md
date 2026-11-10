# Source-data index for figures and Supplementary Tables

This index holds the computational file pointers used to reproduce the article. The manuscript and Supplementary Information retain the scientific definitions, comparisons, units and table numbers. The [variable crosswalk](model_variable_crosswalk.md) explains native field names and analysis labels without changing the original data columns.

All paths below are relative to the repository root. Each figure package contains a `source_data/` directory, caption and reproduction instructions. [paper_assets.json](paper_assets.json) supplies the machine-readable figure-package mapping; [paper_source_data_files.json](paper_source_data_files.json) is the detailed source-artifact file inventory. The documented artifacts must be downloaded or restored before executing their recipes.

## Figure packages

| Article item | Canonical package |
|---|---|
| Fig. 1 | `outputs/age_module/main_figure_redesign/figures/fig01/` |
| Fig. 2 | `outputs/age_module/main_figure_redesign/figures/fig02/` |
| Fig. 3 | `outputs/age_module/main_figure_redesign/figures/fig03_age_mechanism/` |
| Fig. 4 | `outputs/age_module/land_supply_limited_young_retention/figures/retention_candidate/` |
| Extended Data Fig. 1 | `outputs/age_module/main_figure_redesign/figures/ed01_model_architecture/` |
| Extended Data Fig. 2 | `outputs/age_module/main_figure_redesign/figures/ed02_forest_observations/` |
| Extended Data Fig. 3 | `outputs/age_module/main_figure_redesign/figures/ed03_history_age_response/` |
| Extended Data Fig. 4 | `outputs/age_module/main_figure_redesign/figures/ed04_all_pathway_climate/` |
| Extended Data Fig. 5 | `outputs/age_module/main_figure_redesign/figures/ed05_birth_source_response/` |

## Supplementary Tables 1–15

Filenames in the third column are relative to the stated package's `source_data/`, unless a full repository path is given. The tables preserve the original joint-parameter members; reference-year, resolved-peak, follow-up and return flags must be used as defined in the article rather than substituted for one another.

| Supplementary table | Scientific content and source package | Specific records |
|---|---|---|
| Table 1 | Maximum warming and observed overshoot burden; Fig. 1 and Extended Data Fig. 4 | `member_outcomes_all500.csv`, `analysis_summary.json`; five-pathway coverage is in Extended Data Fig. 4 |
| Table 2 | End-of-century carbon and temperature differences; Fig. 2 and the complete core analysis | `annual_members_all500.csv.gz`, `annual_summary.csv`, `analysis_summary.json`; full five-pathway core analysis: `outputs/age_module/main_figure_redesign/analysis/core/` |
| Table 3 | Threshold states and fixed-reference returns; Fig. 1 and Extended Data Fig. 4 | `state_transitions.csv`, `member_outcomes_all500.csv`; Fig. 1 also supplies `target_endpoint_pairs_all500.csv` |
| Table 4 | Delay of return among members returning in both arms; Fig. 1 | `member_outcomes_all500.csv`, `analysis_summary.json` |
| Table 5 | Common four-phase net-uptake contrasts and cumulative-deficit recovery; Fig. 2 | `common_four_stage_net_summary.csv`, `common_four_stage_carbon_members.csv`, `common_four_stage_membership_all500.csv`, `member_stage_carbon_all500.csv`, `analysis_summary.json` |
| Table 6 | Source-selective temperature contrasts and following 20-year growth; Extended Data Fig. 5 | `source_display_statistics.json`, `source_four_arm_members_all500.csv`, `source_member_contrasts_all500.csv`, `source_event_aligned_growth_summary.csv` |
| Table 7 | Cumulative net-uptake effects of birth-source interventions; Fig. 3 and Extended Data Fig. 5 | Fig. 3: `source_member_cumulative_net_endpoints.csv`, `source_cumulative_endpoint_summary.csv`; Extended Data Fig. 5: complete source contrast tables and `source_analysis_summary.json` |
| Table 8 | Production changes and compensating process losses; Fig. 3 | `source_production_to_net_component_summary.csv`, `source_production_to_net_paired_member_cumulative_fluxes_all500.csv`, `source_production_to_net_source_definitions.json` |
| Table 9 | Climate and net-uptake effects of all retention cutoffs; Fig. 4 | `condition_retention_summary.csv`, `member_retention_outcomes_all500.csv` |
| Table 10 | Retained forest area and gross unexecuted activities; Fig. 4 | `condition_retention_summary.csv`, `annual_age_area_contrast.csv.gz`, `summary.json` |
| Table 11 | Secondary standing-harvest delivery and footprint; Fig. 4 | `condition_retention_summary.csv`, `member_retention_outcomes_all500.csv`, `summary.json` |
| Table 12 | Retention and observed return outcomes; Fig. 4 | `return_state_transitions.csv`, `member_retention_outcomes_all500.csv`, `condition_retention_summary.csv` |
| Table 13 | Native regional support | `oscar/_core/_regions/OSCAR_reg_dict.csv`; human-readable coordinate mapping in [model_variable_crosswalk.md](model_variable_crosswalk.md) |
| Table 14 | Conditional removal of the future other-living-tissue establishment ramp | `outputs/age_module/establishment_response_sensitivity/analysis/condition_sensitivity_summary.csv`, `member_establishment_contrasts_all500.csv`, `arm_members_all500.csv`, `summary.json` in the same analysis directory |
| Table 15 | Conditional perturbations to future woody growth rate | `outputs/age_module/establishment_response_sensitivity/k_rate_conditions/analysis/condition_rate_summary.csv`, `member_rate_contrasts_all500.csv`, `arm_members_all500.csv`, `summary.json` in the same analysis directory |

## Additional notes and input provenance

- **The common 2022 atmospheric anchor (Supplementary Note 1).** Fig. 1 `annual_summary.csv` retains the historical annual-mean temperature at 2022. Absolute member-wise atmospheric CO₂ is calculated from `outputs/age_module/historical_emissions/H1/annual_summary.nc` and the corresponding supplied preindustrial CO₂ parameter before ensemble summarisation. The original joint parameters are retained in `inputs/ensemble/OSCARv4_Par_RCMIP3.nc`; `inputs/ensemble/ordered_members.csv` provides identity and order. The prepared archive also preserves the upstream parameter-file identity.
- **Idealised curves and direct landscape production (Supplementary Note 3).** Fig. 3 additionally supplies `source_local_display_curves.csv`, `source_all_region_curve_summary.csv`, `source_landscape_production_geometry.csv` and the member-resolved source files listed in its source-data README. Local establishment, direct production adjustment and complete-land net uptake are different quantities; none is silently substituted for another.
- **Growth inputs and regional reduction (Supplementary Note 5).** The main prepared growth curve is `data/processed/regional_forest_age_inputs_v1/growth_curve.nc` within the native-input bundle. [prepared_inputs_files.json](prepared_inputs_files.json) records the complete prepared archive and its paths. Coefficients, derivatives, interpolation support and preparation metadata are retained with that bundle.
- **Conditional other-tissue experiment (Supplementary Note 6).** The experiment root `outputs/age_module/establishment_response_sensitivity/` preserves `PLAN.json`, `source_masks.json`, `execution_manifest.json` and its analysis. These distinguish treatment X from the reused 00, 10 and 11 arms and preserve the original follow-up masks and unresolved cases. The packaged supplementary source-data artifact uses the corresponding `si_establishment_response_sensitivity/` name.
- **Conditional growth-rate experiments (Supplementary Note 7).** The `k_rate_conditions/` subdirectory preserves its own `PLAN.json`, `source_masks.json`, `execution_manifest.json` and analysis; the prepared-curve provenance records the marginal standard-error input, unchanged spatial support and weights. The packaged supplementary source-data artifact uses `si_establishment_rate_conditions/`. These two conditions remain separate from the parameter-member range.

All exact run names, field names and file locations in this document serve reproducibility. They do not replace the scientific definitions in Methods and the Supplementary Information. See [README.md](README.md) for the ordered model, analysis and figure recipes. Figure rendering reads the supplied source packages and does not launch model experiments.

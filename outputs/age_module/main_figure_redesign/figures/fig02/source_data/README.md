# Source Data

Scientific tables, reduced NetCDF sources and source manifests support this figure. The manifests identify the model runs and prepared inputs used in the analysis.

- [actual_age_stages_all500.nc](actual_age_stages_all500.nc)
- [analysis_summary.json](analysis_summary.json)
- [annual_members_all500.csv.gz](annual_members_all500.csv.gz)
- [annual_summary.csv](annual_summary.csv)
- [common_four_stage_age_means.csv](common_four_stage_age_means.csv)
- [common_four_stage_carbon_members.csv](common_four_stage_carbon_members.csv)
- [common_four_stage_membership_all500.csv](common_four_stage_membership_all500.csv)
- [common_four_stage_net_summary.csv](common_four_stage_net_summary.csv)
- [common_stage_source_manifest.json](common_stage_source_manifest.json)
- [member_outcomes_all500.csv](member_outcomes_all500.csv)
- [member_stage_carbon_all500.csv](member_stage_carbon_all500.csv)
- [stage_summary.csv](stage_summary.csv)

## Event and summary support

All phase boundaries follow each member's age-on events. Saved annual labels partition as y<u, u≤y≤p, p<y<r and y≥r. Only labels 2023–2099 enter uptake integrals; 2022 is a temperature anchor. The common-four-stage display uses the unmodified VL 158/LN 127 mask and the stored `common_four_stage_*` tables. Entire-calendar panels retain all original 500, including missing/censored phase identities. Left-truncated and right-censored cases remain in the complete E0 sources. Phase totals use each member's duration; marginal phase medians must not be summed. Net uptake is the all-native-land anomaly D_Fland−D_Eluc, excluding permafrost, with no allocation of mixed pools by age-area or NPP shares.

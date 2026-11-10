# Source Data

Canonical scientific analysis: `outputs/age_module/land_supply_limited_young_retention/analysis/`. The tables and metadata here derive from that analysis; `summary.json` records the experiment plans and original run identities.

| File | Figure role |
|---|---|
| member_retention_outcomes_all500.csv | All 4000 original-ID rows, exact member pairing, policy/baseline quantities, event/return flags and auxiliary age-off comparisons |
| condition_retention_summary.csv | Canonical within-condition medians and empirical 5–95% ranges; used in panels c–d |
| return_state_transitions.csv | Complete three-state transitions, no-observed-overshoot kept distinct from returned and unreturned |
| summary.json | Completed scientific coverage, plan and original run identities |
| annual_age_area_contrast.csv.gz | Reproducible display derivative: all 8 conditions × 77 native years × 10 saved age/land supports, mean policy/baseline area and their difference |

## Definitions

The main comparison is policy minus the unchanged, fully matched age-on future from the common H1 end-2022 state. Conditions are not pooled. `effect_cumulative_net_uptake_PgC` integrates D_Fland minus D_Eluc over the 77 native 2023–2099 intervals, excluding permafrost; 2022 is not accumulated. Mixed soil, product and other land fluxes are not allocated to age groups by production shares.

Panel b uses the age-resolved forest classes under cutoff 40. Primary is unaged and unchanged and nonforest is not an age class; both are retained in the display table rather than placed on a fabricated age continuum. Classes are [0,5), [5,10), [10,20), [20,30), [30,40), [40,60), [60,100), [100,infinity). Age areas are duration-weighted saved annual support, not an instantaneous final_state inventory. State differences are not tracked cohort flows. Geometry is determined by inputs and policy, so no artificial area uncertainty or jitter is drawn.

Panel c uses every original member's `effect_cumulative_net_uptake_PgC` and `remaining_window_max_K_vs_ageoff`. The baseline reference is `original_age_response_window_max_K`. These are differences of 2022–2099 window maxima, including the common 2022 anchor, not same-year temperature contributions or resolved physical peaks for every member. Its vertical asinh scale has a 35 mK linear width, and every finite signed value is retained. Connected open marks join separate condition medians, not individual member trajectories or an additive budget.

Panel d flips the sign of `effect_Q_achieved_PgC` to show lost secondary standing-harvest delivery of above-ground wood carbon (`rf_sec_agb_achieved`), not total harvested live carbon or all wood services. Only its parameter spread is drawn. `policy_nonforest_gross_unmet_Mha` is cumulative unexecuted gross activity; `policy_nonforest_to_forest_unmet_Mha` is its forest-establishment subset. The latter must not be added again to the former. Gross event areas can revisit land and are not unique preserved hectares or crop yields. Standing harvest retains birth and forest area; its footprint is not forest loss.

Temperature uses the common member-wise 1850–1900 reference. A returned member has an observed exceedance followed by a final annual suffix at or below 1.5 K through 2099; no observed overshoot is separate. Unknown return is not assigned 2099. Age-on reference-returner populations are VL171/LN129; auxiliary age-off reference-returners are VL190/LN138 and are not interchangeable. The caption's unreturned shares use the entire original ensemble, not either conditional denominator.

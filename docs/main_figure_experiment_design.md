# Scientific experiments supporting the paper figures

This document describes completed evidence and its reproducible scientific configuration. The paper uses four main figures and five Extended Data figures, mapped in `reproduction/paper_assets.json`. The experimental families and their scientific roles are mapped in `reproduction/experiments.json`.

## Common model and ensemble

The regional forest-age module uses eleven native land regions and annual secondary/plantation birth cohorts, alongside unaged primary forest and explicit nonforest land classes. Actual birth times, inventories and event routing remain dynamic when the age-production response is disabled. Production-on uses the configured wood and fine-tissue age multipliers; production-off sets both multipliers to one. It does not freeze carbon stocks, mature new forests instantaneously or convert H0 into original OSCAR.

All global experiments retain the supplied 500 joint parameter configurations in their original ID/order. Integration uses native imex, two substeps per year and float64. Historical annual output spans 1750–2022 after a dynamically simulated 850–1750 prehistory; all global futures integrate 77 annual intervals labelled 2023–2099. Label y represents [y−1,y], with an event cohort born at y−0.5. A 2022 row in a future file is an initial anchor, not an additional integrated future year.

Dynamic spin-up begins with primary forest and nonforest land under the fixed preindustrial environment. H0/H1 histories evolve independently from their common accepted spin-up. Their basic futures inherit their own history and the matching atmospheric mode. Emissions-driven futures solve atmospheric CO2 and temperature; prescribed-concentration counterparts constrain CO2 and are retained separately.

## Main climate and carbon comparison

The five complete CMIP7 futures are H, HL, M, VL and LN, each with its matching land activity, wood-carbon requests and atmospheric drivers. The principal paired intervention starts from each member's complete H1 emissions-driven end-2022 restart. The age-on future is reused from `outputs/age_module/future_emissions/<scenario>/H1/`; the future-production-off arm is in `outputs/age_module/future_age_response/<scenario>/off_from_H1_2022/`. They share the same full initial state and drivers. This comparison isolates the future production-response intervention conditional on the historical age-structured state; it is not the separately simulated H1–H0 historical difference.

Temperature uses the common member-wise H1 historical 1850–1900 mean. Window maximum uses 2022–2099 annual temperatures, with the last tied maximum selected. An endpoint maximum is unresolved as a physical peak. Upcrossing, suffix return at or below 1.5 °C, right censoring and missing/never-exceeded states are retained separately. Threshold years and degree-years integrate only the 77 observed future annual intervals; they are not an unobserved complete overshoot duration. Stage comparisons use the same age-on member calendars across their paired carbon/age data; the complete-four-stage display has its explicit support.

Figures 1–2 show climate outcomes, actual forest-age composition, net land uptake and accumulated uptake differences. VL/LN display the overshoot trajectory; all five pathways remain in the complete Source Data and ED4. Parameter-ensemble ranges and pathway differences are reported separately. Net uptake is the actual atmospheric land-budget term D_Fland−D_Eluc; NPP is a distinct production quantity.

## Process and birth-source evidence

Detailed process diagnostics for the five age-on pathways record age-process output at the consumed integration steps, using the same scientific configurations as the matched futures. Annual summaries retain age supports [0,5), [5,10), [10,20), [20,30), [30,40), [40,60), [60,100), 100+ and unaged primary. Fig. 2b groups these into <5, 5–20, 20–40, 40–100 and ≥100 years plus unaged primary forest; historical ED3 retains its own stated grouping. True establishment, primary-to-secondary renewal, secondary conversion exits and standing wood removals are distinguished. Secondary standing harvest retains birth time and area under the native contract; continuing birth does not imply that every young stand is repeatedly clear-cut.

In VL/LN, the source-selective 2×2 applies age-production responses to either cohorts already present in 2022 or cohorts born later. Existing-only and post-2022-only configurations use the matched all-off/all-on arms as references. All four share the same full H1-2022 restart and complete pathway inputs. Conditional source effects and their interaction retain the original joint population; they do not allocate mixed soil or product pools by NPP shares.

The local L1–L4 experiments use the formal carbon/event kernel over 0–100 relative years under the fixed 2000–2019 calibration environment. They compare one-hectare establishment once or in twenty equal annual increments, with production-on/off, in all eleven regions and all original members. Both schedules start with the same H1-2022 per-area pasture (`pastr`) living carbon and Pasture-biome CWD/soil; the product pool is a shared normalized background, not an inferred pasture carbon ancestry. Unestablished land retains its nonforest process; cumulative established area is one hectare, whereas forest area-years differ. Products are not emitted instantaneously. These are local fixed-environment responses, not global climate scenarios.

Figure 3 combines these local responses, actual landscape age/entry dynamics and the completed birth-source net-uptake intervention. Production and net uptake remain distinct. A full-window production-to-net bridge identifies available actual loss terms without calling an unidentified arithmetic residual an age-specific sink.

## Supply-limited young-forest retention

Eight age-on configurations use complete VL/LN inputs and the common H1-2022 restart, with dynamic protection below 10, 20, 30 or 40 actual years. The baseline current-state secondary conversion and standing-harvest proposals are formed first; the protected portions are cancelled without replacement by older, primary or other-region forest. A stand becomes eligible when it reaches the cutoff. Primary forest is unaged and its original activity is not increased. Natural mortality/fire are not eliminated; no windthrow mechanism is asserted.

Retained forest can reduce future nonforest supply. In every region/year/nonforest source c, all its original outgoing gross requests share alpha_c=min(1,A_c/sum(request_c)), where A_c is opening source area before current-year incoming. Actual activities, destinations and that source's harvest are scaled by the same factor; original requests and unmet activity remain explicit. This includes reduced nonforest-to-forest establishment. Standing-harvest delivery/footprint and wood-carbon shortfall are recorded independently. The result is a management scenario with changed land/wood services, not equal-service or a pure age-response effect. The strict-request retention cases are area-infeasible and have no coupled climate result.

Figure 4 compares these eight formally completed configurations to their unchanged age-on baselines and retains the auxiliary age-off climate reference. It does not infer an optimal protection age or a technical/economic value for service losses.

## Observation and historical context

ED2 compares the current emissions-driven H1 2020 state with existing GAMI age and FRA2025 forest-area/living-carbon observations on their stated eleven-region supports, retaining missingness and differing forest denominators. It does not invent grid-scale model age maps or assign primary forest a numerical age. ED3 shows independently evolved H0/H1 historical age, stock, net-exchange and atmospheric differences; concentration and emissions modes are not pooled. ED1 describes the architecture and ED5 the conditional birth-source responses.

## Reproduction

`reproduce.py` provides explicit recipes; `inputs/README.md` identifies prepared sources and the unchanged dedicated 500-member file. The code licence remains upstream CeCILL 2.1. Scientific plans and model reports record the implementation and inputs for each experiment. Prepared inputs and paper Source Data are distributed as release assets. The plotting recipes consume these sources; model integrations require the explicitly ordered preceding model recipes.

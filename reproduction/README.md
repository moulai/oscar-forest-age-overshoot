# Reproducing the study

Run all commands from the checkout root using the recorded environment. Download the v1.0.0 prepared-input and Source Data assets linked from the repository README before checking a data-dependent recipe. `reproduce.py` reads `commands.json`; it is a small explicit command dispatcher, not an automatic scientific workflow or scheduler. It never launches anything on import or on `--list`, `--show` or `--check`.

The [source-data index](source_data_index.md) maps the figures and Supplementary Tables to their files. The [model-variable crosswalk](model_variable_crosswalk.md) maps the manuscript's physical descriptions to preserved native fields and analysis columns.

```bash
.venv/bin/python reproduce.py --list
.venv/bin/python reproduce.py --check paper-figures
.venv/bin/python reproduce.py --show paper-figures
.venv/bin/python reproduce.py --run paper-figures
```

## Three levels

1. **Re-render the nine paper figures.** Supply the current figure Source Data packages and captions, then run `paper-figures`. These commands write to `outputs/reproduced/figures/` and do not modify the delivered figure packages or invoke a model. `paper_assets.json` gives the canonical nine source packages. The delivered source artifact, including the compact core analysis tables, supports these routes; their scientific scope remains the original five-pathway or fixed-subset scope.
2. **Rebuild the paper analyses.** `paper-analyses` lists the completed model outputs required for the core, detailed age stages, production-to-net bridge, retention intervention and historical/observation comparisons. Commands write separate reconstructed analyses; `paper-figures-from-analysis` then renders all nine from those new analyses. `matched-future-analysis` and `birth-source-analysis` regenerate the compact paired sources from their formal experiment outputs. All pairing uses the same original IDs before summarizing; no conditions are pooled as independent observations.
3. **Reintegrate the experiments.** The `model-*` recipes show the actual scientific runners, original inputs and dependencies. Use a fresh data checkout or an explicitly separate result root; scientific runners refuse to replace existing complete experiment directories. Run preparation before the corresponding integration to generate a fresh implementation record. Each PLAN records the implementation and settings used for its experiment. Prepared inputs are sufficient for the model route; raw-data rebuilding is separately described in `inputs/`.

## Scientific order for the main evidence

Prepared inputs → dynamic primary-forest spin-up → independent H0/H1 prehistories/histories → each arm's matched CMIP7 futures → common-H1-2022 future-production-off comparison (on reused) → source-selective VL/LN responses. Detailed process diagnostics use the same five age-on scientific configurations. The local one-hectare idealized experiments and the explicit supply-limited retention experiment answer different questions and keep separate identities. Paper analyses and graphics then consume these completed sources.

## Finite establishment-response conditions

These six additional matched VL/LN configurations reuse completed full-response (11), existing-response-only (10) and production-unmodulated (00) results. Each new condition keeps the same original ordered 500 members, full H1 end-2022 state, complete pathway drivers, gross event requests and numerical scheme. Actual inventory, climate and standing-harvest delivery/worked area remain coupled.

| Explicit recipe | New configurations | Consumed production change | Output root |
|---|---|---|---|
| `model-establishment-f-response` | VL/LN: 2 | Only post-2022 cohort F multiplier becomes 1; original W and existing F remain | `outputs/age_module/establishment_response_sensitivity/` |
| `model-establishment-k-rate` | VL/LN × k±reported marginal SE: 4 | Only post-2022 W uses the alternative S and S′; existing W, every F, alpha and calibration lambda remain | Previous root plus `k_rate_conditions/` |

Run these explicit recipes only after `model-birth-source` in a fresh canonical result tree. They do not rerun the reused baselines and do not cross the F and k treatments. Each runner prepares its own immutable scientific PLAN, performs a no-integration multiplier-interface check, integrates its new conditions, and creates the paired analysis automatically. The recorded original-member solver is float64/imex with two steps per year and no adaptive increase in substeps; worker batches remain ten. The recipes do not execute on `--list`, `--show` or `--check`.

The k inputs are built by `workflows/prepare_robinson_k_error_support.py` and `workflows/prepare_robinson_conditional_k_curves.py`; the frozen prepared-input manifest and centre-isolation record are at `outputs/01_prepared_data/robinson_conditional_k_v1/manifest.json`. The original valid support and area/capacity weighting are retained. These coherent one-factor conditions are not confidence bounds, independent NPP validation or a complete structural-uncertainty envelope.

The F analysis reports new−10 and 11−new. The k analysis reports Yk−11 and Yk−10, with additional total contrasts to 00. All differences are computed per member before summarising. Common-calendar comparisons retain each 00 maximum year and the original +20-year eligibility, separating resolved overshoots. The complete net-land balance is D_Fland−D_Eluc, excluding permafrost; native2023–2099 integration excludes the common2022 climate anchor. Supplementary source data retain the member-level contrasts and flags, not an additional figure.

## Fresh numerical reproduction, in order

Run these explicit recipes sequentially in a fresh canonical result tree: `model-spinup`, `model-histories`, `model-matched-on`, `model-matched-off`, `model-birth-source`, `model-establishment-f-response`, `model-establishment-k-rate`, `model-detailed-process`, `model-idealized`, `model-retention`, `paper-analyses`, then `paper-figures-from-analysis`. Each recipe can be inspected with `--show` before `--run`. The on/off runners automatically build their own canonical paired thin analyses; `paper-analyses` derives the paper tables in the separate reconstruction output. English captions and the small architecture connection table are versioned with the code. The figure recipes require Arial.

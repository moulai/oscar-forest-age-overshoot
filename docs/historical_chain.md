# Historical spin-up and H0/H1 chain

The historical entry points are `workflows/run_regional_forest_age.py` and `workflows/run_regional_forest_age_batches.py`. Spin-up is `--phase spinup` on the first script. H0 and H1 are the batch script, which launches one history phase per member slice and merges the original 500 members back into their original order. Both history arms read `workflows/prepare_regional_forest_age_inputs.py` for `load_runtime_inputs`; that import does not rebuild the inputs.

The runners use command-line settings rather than a JSON configuration file:

| Run | age production | tau (years) | harvest contract | batch size |
| --- | --- | --- | --- | --- |
| Spin-up | off | 0 | not used | full ensemble, one process |
| H0 | off | 2 | `secondary_gross_agb_request_proportional` | 5 |
| H1 | on | 2 | `secondary_gross_agb_request_proportional` | 1 |

History selection is `area_proportional`. Substeps per year (`--nt`) are 2. Spin-up uses the script defaults of 100-year windows, a 6000-year ceiling, and local density tolerances `1e-7` PgC/Mha and `1e-5` relative. Worker count is not a scientific setting.

## Paths opened at run time

Spin-up opens:

- `data/processed/regional_forest_age_inputs_v1/manifest.json`
- `data/processed/regional_forest_age_inputs_v1/primary_parameters.nc`
- `data/processed/regional_forest_age_inputs_v1/calibration_environment.nc`
- `data/processed/regional_forest_age_inputs_v1/land_metadata.nc`

It then integrates preindustrial forcing generated in memory. It does not open a forcing file, a growth curve, or a secondary-harvest file.

H0 and H1 additionally open:

- `data/processed/regional_forest_age_inputs_v1/prehistory_forcing.nc` (850–1750)
- `data/processed/regional_forest_age_inputs_v1/historical_forcing.nc` (1750–2022)
- `outputs/age_module/spinup/report.json`
- `outputs/age_module/spinup/baseline_parameters.nc`
- `outputs/age_module/spinup/final_state.nc`
- `data/processed/regional_secondary_harvest_carbon_v1/manifest.json`
- `data/processed/regional_secondary_harvest_carbon_v1/prehistory_requests.nc`
- `data/processed/regional_secondary_harvest_carbon_v1/historical_requests.nc`
- `data/processed/regional_secondary_harvest_carbon_v1/projection_parameters.nc`

H1 also opens `growth_curve.nc` and `wood_calibration.nc` in the same inputs directory, because age-dependent production is on. H0 and the spin-up do not open those two files. They are present so one input directory serves all three runs.

The history phase checks that the spin-up report status is `passed_dynamic_spinup`, that its input-manifest SHA-256 matches `manifest.json`, and that the spin-up had no birth slots. The copied manifest hash is `0fce80afb4d96b7f070a1c9bdab8be79f59dd9c424eb9064733616ffed2266b8`.

`pi_environment.nc` and `source_history.nc` are not opened by the integrator. They are kept because the prepared-input manifests name them and record their hashes. Future forcing files named by the native input manifest are not used by spin-up or by H0/H1 and are not part of `data/processed/` here.

The runners also call `git rev-parse HEAD` and record that commit. They do not read raw rasters, masks, or scenario libraries at run time.

## Packaged input copies

`outputs/01_prepared_data` contains a packaged copy of the native inputs, the secondary extension, and the reduced GAMI file. The commands above use `data/processed/` so the relative paths match the historical runs. The reduced 2010/2020 observations used by the age comparison are only the packaged file:

`outputs/01_prepared_data/native_inputs_v1/native_inputs_v1/results/gami_cci_reduction/global_v1/observations.nc`

The original 500-member parameter file inside that package is `data/raw/parameters/provided_500/OSCARv4_Par_RCMIP3.nc`, under the same `native_inputs_v1` prefix. The historical runner reads `primary_parameters.nc`, which already carries that ensemble in order. The same original parameter file is versioned in `inputs/ensemble/`.

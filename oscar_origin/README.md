# Original OSCAR

This tree is upstream OSCAR commit `cb368fc09d21567dfc8de60f3507694838f9d4f5`, plus exactly four confirmed fixes. It contains none of the forest-age module.

Set `PYTHONPATH` to this directory to import the package name `oscar`. The age-module model remains the sibling directory `../oscar` and is still the default `import oscar` from the study root.

## Fixes

1. `Eq__ERF_lap` copies `D_Ebb_BC` with `.copy(deep=True)` before the in-place subtraction.
2. `Eq__Eimb_CO2` adds `D_Foxi_CH4`.
3. `safe_exp` is the stable `logaddexp` form from commit `3b731ad`.
4. N2O equations are unchanged. The mass-basis correction stays in the shared prepared drivers under `data/processed/`, not in `OSCAR_N2O.py`.

`OSCAR_landC.py` is the upstream bookkeeping version. The land-step changes in `cls_main.py` are not included.

# Original forest support for the optional k-error reduction

`forest850_region_weights.nc` is the unchanged project-local copy of the original LUH850 forest support and native regional allocation. It retains the source forest area, conditional regional fractions, and spherical fine-row subdivision used for the central regional curves. Its provenance is recorded in `source_manifest.json`; the NetCDF is excluded from Git.

Run `.venv/bin/python -u -B workflows/prepare_robinson_k_error_support.py --phase coverage` from the project root after acquiring the four official Robinson v1 rasters. The summary and regional CSV are written to `outputs/01_prepared_data/robinson_k_error_support_v1/`. The coverage output is an input description: it does not generate a perturbed curve or integrate the model.

Forest-area and recoverable-capacity denominators are distinguished. Capacity is the original raster quantity A[1-(1-b)^3], not model NPP or the model forest-carbon inventory. Missing/negative SE, nonpositive k-SE and large relative errors remain explicit. Weighted quantiles use a single-pass histogram and retain their bin bounds.

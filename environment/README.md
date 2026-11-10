# Recorded computational environment

The completed study used CPython 3.10.12 on Linux x86-64. `requirements-lock.txt` records the Python package versions used by the study. The scientific code is imported from this checkout with `PYTHONPATH=.`. Installing an unrelated PyPI package named `oscar` is not required.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements-lock.txt
export PYTHONPATH=.
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python reproduce.py --list
```

The original-OSCAR recipes set `PYTHONPATH=oscar_origin` for their own processes; the age-module recipes use the checkout root. The reproduction entry does this explicitly. `numpy`, `xarray`, `netCDF4`, `h5netcdf`, `scipy`, `pandas`, `matplotlib`, `PyYAML`, `requests` and Pillow support the current scientific and figure workflows. No R environment, Office application or GPU is required. TIFFs use Pillow's lossless LZW export.

The paper plotting scripts require Arial in regular, bold, italic and bold-italic styles and stop if it is unavailable. Install a licensed copy of Arial before plotting. Font files are not redistributed with this code. The publication bundles retain their exact rendered outputs and source scripts. PDF/SVG preserve text; formal PNG/TIFF are 300/600 dpi, while the separate final-size reading preview is 96 dpi.

Model workers use one BLAS/OpenMP thread each. The recorded scientific contract is `nt=2`, imex integration, float64 and hash seed zero. Worker count controls resource use rather than the selected member population. Do not launch several high-concurrency model recipes simultaneously; the completed study used a total budget of approximately 500 single-thread workers.

The optional Robinson v1 marginal-standard-error input reduction additionally uses `requirements-input-preparation.txt` (Rasterio 1.4.3 and its dependencies) for raster acquisition and coverage summaries. Install both requirement files together when preparing these optional inputs.

# Inputs and their acquisition

Inputs are separate from model outputs even where an inherited path starts with `outputs/01_prepared_data/`. Existing canonical paths are preserved to retain the recorded scientific identities. `catalogue.json` lists roles, source manifests, preparation code and the prepared files used by the study. The original dedicated parameter file is in [ensemble/](ensemble/README.md) and is included in this repository.

## Reproduce from prepared artifacts

Extract the prepared-input artifact at the checkout root to populate the relative paths below.

| Relative location | Required contents and use |
|---|---|
| `data/processed/regional_forest_age_inputs_v1/` | Primary parameters, land-event metadata, growth and wood calibrations, fixed calibration environment, 850–1750 prehistory and 1750–2022 historical forcing |
| `data/processed/regional_secondary_harvest_carbon_v1/` | Historical/prehistory secondary gross-carbon requests and the operational aboveground projection |
| `outputs/01_prepared_data/cmip7_n2o_mass_corrected_v1/` | Corrected historical atmosphere, five matched future forcings and requests, original-OSCAR drivers and indexed parameters |
| `outputs/01_prepared_data/cmip7_n2o_mass_corrected_H0_v1/` | The corresponding H0-bound continuation inputs used by the basic futures |
| `outputs/01_prepared_data/native_inputs_v1/native_inputs_v1/` | Original ordered ensemble, reduced GAMI observation support and immutable parent provenance |
| `outputs/01_prepared_data/robinson_conditional_k_v1/` | Fixed-support central and k±reported-SE regional curves, both original weightings, node table and frozen source manifest; conditional supplementary inputs, not a joint uncertainty interval |
| `outputs/01_prepared_data/secondary_candidate_inputs_v1/secondary_candidate_inputs_v1/` | FRA2025 tables, source provenance and secondary-carbon add-on records |

The current [reproduction guide](../reproduction/README.md) defines the experiment settings. Dataset versions, source checksums and scientific definitions remain in the input manifests. Historical machine-specific path prefixes are represented by portable provenance identifiers in the public assets; these identifiers are source records, not runtime paths. The asset transformation manifest records original and public file digests.

## Obtain original products or rebuild prepared inputs

The parent manifests under the two input bundles preserve file-level download URLs, provider versions, units, checksums and processing identities. LUH3 historical v1.2 is recorded at DOI [10.5281/zenodo.19261724](https://doi.org/10.5281/zenodo.19261724). Future LUH3 and CMIP7 concentration/emission sources retain their distinct source IDs and versions in `manifests/future_land_sources_v1.json` and `manifests/future_input_sources_v1.json`. The recorded original OSCAR CMIP7 library is [Zenodo record 22709506](https://zenodo.org/records/22709506). GAMI, FRA and other observational records are mapped in the catalogue; the current observation figure uses GAMI age and FRA forest area/living carbon, rather than treating input calibration data as independent future-climate validation.

`inputs/preparation/README.md` identifies the input-preparation routines and their dependencies. Reconstructing raw raster reductions requires the separately documented public products, not a previous forest-age simulation. Running the study from its prepared inputs requires no raw 100-m rasters.

Prepared inputs are distributed in [prepared_inputs.tar.gz](https://github.com/moulai/oscar-forest-age-overshoot/releases/download/v1.0.0/prepared_inputs.tar.gz); figure and Supplementary Source Data are in [paper_source_data.tar.gz](https://github.com/moulai/oscar-forest-age-overshoot/releases/download/v1.0.0/paper_source_data.tar.gz). Extract them at the repository root. The full trajectory archive can be regenerated with the explicit model recipes. The original 500-member file and its member-order table are included directly in Git.

The optional conditional-k inputs retain the unchanged forest support in `inputs/robinson_k_error_support/forest850_region_weights.nc`. Their official Robinson v1 raster URLs, version and verified acquisition are listed in `inputs/raw/Robinson2025_v1/source_manifest.json`; raw GeoTIFFs are excluded from Git. `workflows/prepare_robinson_k_error_support.py` describes the SE coverage, and `workflows/prepare_robinson_conditional_k_curves.py` rebuilds the three small runtime curves with the original regional reduction and interpolation. Raster preparation additionally requires `environment/requirements-input-preparation.txt`.

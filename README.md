# Forest age and climate overshoot

Scientific code for *Lagged carbon uptake in new forests raises the overshoot peak and delays the return to 1.5 °C*.

The study extends OSCAR v4, a compact earth system model, with native regional annual forest cohorts and age-dependent production. Paired experiments examine changes in land-carbon uptake, peak warming and returns to 1.5 °C in five matched CMIP7 pathways. Cohort-attribution experiments, idealized establishment experiments, establishment-response sensitivities and protection of young secondary forest provide the process and intervention evidence.

## Installation

```bash
git clone https://github.com/moulai/oscar-forest-age-overshoot.git
cd oscar-forest-age-overshoot
python3.10 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements-lock.txt
```

This repository release is v1.0.0; `VERSION` records the underlying OSCAR model version, v4-beta2. Use the recorded [environment](environment/README.md). Model integrations use the local `oscar/` package; do not install an unrelated package with the same name. The original 500-member joint parameter file, 327 parameter variables and ordered source IDs are included in [inputs/ensemble/](inputs/ensemble/README.md).

## Data and reproduction

The [v1.0.0 data assets](https://github.com/moulai/oscar-forest-age-overshoot/releases/tag/v1.0.0) contain the prepared inputs and the current figure and Supplementary Source Data. They are distributed separately from Git. Download and extract them **at the checkout root**:

```bash
curl -fL -o prepared_inputs.tar.gz https://github.com/moulai/oscar-forest-age-overshoot/releases/download/v1.0.0/prepared_inputs.tar.gz
curl -fL -o paper_source_data.tar.gz https://github.com/moulai/oscar-forest-age-overshoot/releases/download/v1.0.0/paper_source_data.tar.gz
tar -xzf prepared_inputs.tar.gz
tar -xzf paper_source_data.tar.gz
.venv/bin/python reproduce.py --list
.venv/bin/python reproduce.py --check paper-figures
.venv/bin/python reproduce.py --show paper-figures
.venv/bin/python reproduce.py --run paper-figures
```

The final command redraws four main figures and five Extended Data figures in `outputs/reproduced/figures/`. It reads delivered Source Data and does not run model integrations. The published graphics require Arial; install a licensed Arial font before plotting. Data-asset sizes and SHA-256 digests are recorded in [reproduction/data_artifacts.json](reproduction/data_artifacts.json).

The [reproduction guide](reproduction/README.md) separates redrawing figures, rebuilding analyses and fresh model integrations. Model recipes require prepared inputs and the explicitly ordered preceding experiments; they do not download data or execute automatically. Full historical and future trajectory archives are not part of the compact figure asset. They can be regenerated with the model recipes. All comparisons retain original member IDs, order and joint parameters.

## Directory layout

| Directory | Contents |
|---|---|
| `oscar/` | OSCAR and the native forest-cohort implementation |
| `oscar_origin/` | Original OSCAR reference implementation and its attribution |
| `workflows/` | Input preparation, scientific runners, analyses and nine-figure plotting code |
| `inputs/` | Acquisition methods, source catalogue, ensemble and preparation provenance |
| `environment/` | Recorded scientific and optional raster-preparation dependencies |
| `docs/` | Scientific experiment, historical-chain and protection definitions |
| `reproduction/` | Explicit recipes, figure/table mappings and data-asset manifests |
| `tests/` | Small tests of cohort, event and numerical interfaces |
| `data/processed/` | Prepared historical inputs, populated by the input asset |
| `outputs/01_prepared_data/` | Prepared CMIP7/observation inputs, populated by the input asset |
| `outputs/age_module/` | Scientific outputs and figure Source Data at their canonical relative paths |

The input path `outputs/01_prepared_data/` is retained for compatibility with the scientific recipes. Inputs and simulation outputs have distinct roles in the [input catalogue](inputs/catalogue.json).

## Scientific interpretation

Production off sets the specified age multipliers to one while forest age, stocks and events continue to evolve. The main future comparison shares each member's full historical end-2022 state; the H0 and H1 historical runs evolve independently. Net land uptake includes living, debris, soil and product pools and excludes permafrost. It differs from forest NPP and changes in vegetation stocks.

The five complete futures are High (H), High-to-Low (HL), Medium (M), Very Low (VL) and Low-to-Negative (LN). Detailed process attribution has its stated experimental support. Cohort attribution separates stands by birth time, rather than tracing carbon atoms through shared pools. Parameter-ensemble ranges describe the supplied ensemble; member fractions are not calibrated real-world probabilities. Responses of the prescribed production–age relationships remain conditional on their ecological assumptions.

## Citation, provenance and licence

Citation metadata are in [CITATION.cff](CITATION.cff). The manuscript is not assigned a publication DOI in this release.

The upstream [CeCILL v2.1 licence](LICENSE) and existing author notices are retained. [Licence and source notes](LICENSE_PROVENANCE.md) distinguish the code licence from third-party data terms and the supplied ensemble. [Data provenance](reproduction/DATA_PROVENANCE.md) identifies the products, transformations and licence records associated with the release assets. Cite OSCAR and the individual data products as well as this study.

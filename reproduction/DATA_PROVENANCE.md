# Data provenance and release-asset terms

The release provides prepared regional inputs and study-generated figure and Supplementary Source Data. Provider versions, file URLs, original checksums, scientific definitions and licence records accompany the prepared-input asset. The code licence does not replace these data terms.

| Product | Source/version and use | Recorded terms |
|---|---|---|
| Forest biomass–age curves | Robinson et al. v1, https://doi.org/10.5281/zenodo.15090826; coefficients, standard errors and regional reductions | CC BY 4.0 |
| GAMI forest age | GAMI v2.1, https://doi.org/10.5880/GFZ.1.4.2023.006; aggregated age observations | Provider attribution and terms retained in the source records |
| Forest area and living carbon | FRA2025 platform, https://fra-data.fao.org/; regional 2020 inventories | FAO attribution and platform terms |
| Land-use history | LUH3 historical v1.2, https://doi.org/10.5281/zenodo.19261724 | Provider CC BY 4.0 record and verbatim licence wording retained |
| Future land use | LUH3 v3.1.1 for H/VL and v3.1 for HL/M/LN; https://input4mips-cvs.readthedocs.io/en/latest/dataset-overviews/land-use/ | CC BY 4.0 in source manifests |
| Atmospheric drivers | OSCAR v4 library v4-beta2.1, https://doi.org/10.5281/zenodo.22709506, and the versioned CMIP7 input4MIPs products listed in the manifests | Individual provider records retained; no blanket relicensing |
| ESA CCI biomass support | Regional observation/preparation support; original version and source URLs in the input configurations | https://artefacts.ceda.ac.uk/licences/specific_licences/esacci_biomass_terms_and_conditions_v2.pdf |
| Original parameter ensemble | OSCAR v4-beta2, parameter-data version 2.0.1; 500 ordered joint parameter sets, 327 variables | Provider metadata and original digest retained; no separate data licence was supplied |

The LUH3 source text uses “Share Alike” while linking to the CC BY 4.0 URL. The release preserves both the original wording and the recorded identifier. Raw global rasters are not redistributed in these assets; public retrieval locations and preparation code are provided.

`prepared_inputs.tar.gz` contains the labelled inputs needed for fresh integrations and input/observation provenance. `paper_source_data.tar.gz` contains current nine-figure Source Data, core paired summaries, English captions and the two supplementary establishment-sensitivity source sets. It does not contain the full historical/future trajectory archive.

Each asset includes a `*_DATA_MANIFEST.json` with relative paths, file sizes and SHA-256 digests. Source records identify provider versions, data-processing definitions and the implementation used for each dataset. Recipes resolve inputs from the checkout root using the paths listed in the asset inventories.

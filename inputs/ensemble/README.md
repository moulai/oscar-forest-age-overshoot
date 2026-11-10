# Original joint 500-member ensemble

`OSCARv4_Par_RCMIP3.nc` is the unchanged project-provided parameter file (2,781,537 bytes; 327 data variables, 500 configurations). It contains the joint parameters, not just a list of independent marginals. `ordered_members.csv` maps member position 0–499 to the original config IDs in their supplied order (first 11, last 9993). No resampling, filtering, weighting or parameter recombination is performed.

`manifest.json` retains the original source-file digest, provider/model metadata and order identity. The parameters inside generated model runs also retain `source_config`, so a native contiguous config position cannot be confused with an original ID. The input-bundle copy and the file in `inputs/ensemble/` reference the same original ensemble.

The original record describes a project-provided ensemble with model OSCARv4, model_version v4-beta2, data_version 2.0.1 and Thomas Gasser as the file contact. A separate data licence and the ensemble-generation method were not supplied in that record. See LICENSE_PROVENANCE.md at the repository root.

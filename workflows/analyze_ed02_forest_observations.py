#!/usr/bin/env python3
"""Compare the completed emissions-H1 2020 checkpoint with prepared observations.

This workflow does not fit, calibrate, simulate, or reinterpret unaged primary
forest as a numerical age. Each source retains its original forest support.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REGIONS = ["ROWO", "AFRICA", "CHINA+", "EUROPE", "INDIA+", "LATIN_AM",
           "MIDDLE_EAST", "NORTH_AM", "PAC_OECD", "REF_ECON", "REST_ASIA"]
AGE_CLASSES = ["0_to_20", "21_to_40", "41_to_60", "61_to_80", "81_to_100",
               "over_100_resolved", "primary_unaged", "gami_mature_code300",
               "gami_unresolved_code299"]


def describe(a):
    a = np.asarray(a, dtype=float)
    return {"n": int(np.isfinite(a).sum()), "mean": float(np.nanmean(a)),
            "median": float(np.nanmedian(a)), "q05": float(np.nanquantile(a, .05)),
            "q95": float(np.nanquantile(a, .95))}


def clean_json(v):
    if isinstance(v, dict):
        return {str(k): clean_json(w) for k, w in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean_json(w) for w in v]
    if isinstance(v, (np.integer, np.floating)):
        return clean_json(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def run(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    state_path = root / "outputs/age_module/historical_emissions/H1/checkpoints/state_2020.nc"
    obs_path = root / "outputs/01_prepared_data/native_inputs_v1/native_inputs_v1/results/gami_cci_reduction/global_v1/observations.nc"
    fra_dir = root / "outputs/01_prepared_data/secondary_candidate_inputs_v1/secondary_candidate_inputs_v1/data/processed/fra2025_inventory_v1"
    map_path = root / "oscar/_core/_regions/OSCAR_region_mask_1deg.nc"
    dictionary_path = root / "oscar/_core/_regions/OSCAR_reg_dict.csv"
    input_manifest = json.loads((root / "data/processed/regional_forest_age_inputs_v1/manifest.json").read_text())
    with xr.open_dataset(state_path) as ds:
        ids = np.asarray(ds.source_config.values)
        if ids.tolist() != input_manifest["source_config"]:
            raise ValueError("The checkpoint must retain the original ordered joint-parameter IDs.")
        if ds.reg_land.values.astype(str).tolist() != REGIONS or len(ids) != 500:
            raise ValueError("Unexpected model region or member support.")
        year = float(ds.year)
        if year != 2020:
            raise ValueError("The comparison requires the actual 2020 checkpoint.")
        af = ds.rf_A_forest.transpose("config", "reg_land", "forest_class", "birth_time").values.sum(axis=2)
        age = year - ds.birth_time.values
        if np.any(af[..., age < 0] > 1e-9):
            raise ValueError("An occupied future cohort cannot be assigned an age.")
        primary = ds.rf_A_primary.transpose("config", "reg_land").values
        live = np.zeros_like(primary)
        for name in ["rf_W_forest", "rf_F_forest"]:
            live += ds[name].sum(["forest_class", "birth_time"]).transpose("config", "reg_land").values
        for name in ["rf_W_primary", "rf_F_primary"]:
            live += ds[name].transpose("config", "reg_land").values
        model_identity = ds.attrs["run_identity"]
        model_attrs = dict(ds.attrs)
    # Right-closed 20-year classes follow the published GAMI integer endpoints;
    # no exact age is inferred for primary forest or the GAMI terminal code.
    masks = [(age >= 0) & (age <= 20), (age > 20) & (age <= 40),
             (age > 40) & (age <= 60), (age > 60) & (age <= 80),
             (age > 80) & (age <= 100), age > 100]
    model_age = np.zeros((500, 11, len(AGE_CLASSES)))
    for k, mask in enumerate(masks):
        model_age[:, :, k] = af[:, :, mask].sum(axis=2)
    model_age[:, :, 6] = primary
    area = af.sum(axis=2) + primary
    model_rows, age_rows = [], []
    for j, region in enumerate(REGIONS + ["GLOBAL"]):
        ar = area[:, j] if j < 11 else area.sum(axis=1)
        lv = live[:, j] if j < 11 else live.sum(axis=1)
        aa = model_age[:, j, :] if j < 11 else model_age.sum(axis=1)
        for i, sid in enumerate(ids):
            model_rows.append({"member_position": i, "source_config": sid,
                "region": region, "forest_area_Mha": ar[i], "live_carbon_PgC": lv[i],
                "live_density_MgC_ha": lv[i] * 1000 / ar[i] if ar[i] > 0 else np.nan,
                "known_young_share_total_forest_pct": aa[i, 0] / ar[i] * 100 if ar[i] > 0 else np.nan,
                "primary_unaged_share_total_forest_pct": aa[i, 6] / ar[i] * 100 if ar[i] > 0 else np.nan,
                "known_young_share_aged_forest_pct": aa[i, 0] / (ar[i] - aa[i, 6]) * 100 if ar[i] > aa[i, 6] else np.nan})
            for k, group in enumerate(AGE_CLASSES):
                age_rows.append({"member_position": i, "source_config": sid,
                    "region": region, "age_class": group, "area_Mha": aa[i, k],
                    "share_total_forest_pct": aa[i, k] / ar[i] * 100 if ar[i] > 0 else np.nan})
    model = pd.DataFrame(model_rows)
    model.to_csv(output / "model_regional_members_all500.csv", index=False)
    pd.DataFrame(age_rows).to_csv(output / "model_age_composition_all500.csv.gz", index=False)

    with xr.open_dataset(obs_path) as ds:
        hist = ds.integer_age_area_ha.sel(year=2020, support="gami_native").transpose("member", "region", "integer_age").values / 1e6
        gami_area = ds.gami_area_ha.sel(year=2020).transpose("member", "region").values / 1e6
        greg = ds.region.values.astype(str).tolist()
        gmember = ds.member.values
        gages = ds.integer_age.values
        gami_attrs = dict(ds.attrs)
        gami_identity = ds.attrs["run_identity"]
    gami_age = np.zeros(hist.shape[:2] + (len(AGE_CLASSES),))
    gmasks = [(gages >= 0) & (gages <= 20), (gages > 20) & (gages <= 40),
              (gages > 40) & (gages <= 60), (gages > 60) & (gages <= 80),
              (gages > 80) & (gages <= 100), (gages > 100) & (gages < 299)]
    for k, mask in enumerate(gmasks):
        gami_age[:, :, k] = hist[:, :, mask].sum(axis=2)
    gami_age[:, :, 7] = hist[:, :, gages == 300].sum(axis=2)
    gami_age[:, :, 8] = hist[:, :, gages == 299].sum(axis=2)
    g_rows, ga_rows = [], []
    for j, region in enumerate(greg + ["GLOBAL"]):
        ar = gami_area[:, j] if j < len(greg) else gami_area.sum(axis=1)
        aa = gami_age[:, j, :] if j < len(greg) else gami_age.sum(axis=1)
        for i, mem in enumerate(gmember):
            g_rows.append({"gami_member": mem, "region": region, "forest_area_Mha": ar[i],
                "young_share_total_forest_pct": aa[i, 0] / ar[i] * 100 if ar[i] > 0 else np.nan,
                "mature_code300_share_total_forest_pct": aa[i, 7] / ar[i] * 100 if ar[i] > 0 else np.nan,
                "unresolved_code299_area_Mha": aa[i, 8]})
            for k, group in enumerate(AGE_CLASSES):
                ga_rows.append({"gami_member": mem, "region": region, "age_class": group,
                    "area_Mha": aa[i, k], "share_total_forest_pct": aa[i, k] / ar[i] * 100 if ar[i] > 0 else np.nan})
    gami = pd.DataFrame(g_rows)
    gami.to_csv(output / "gami_regional_members_all20.csv", index=False)
    pd.DataFrame(ga_rows).to_csv(output / "gami_age_composition_all20.csv", index=False)
    # Preserve native integer categories, including both special endpoints.
    xr.Dataset({"forest_area_Mha": (("gami_member", "region"), gami_area),
        "integer_age_area_Mha": (("gami_member", "region", "integer_age"), hist)},
        coords={"gami_member": gmember, "region": greg, "integer_age": gages},
        attrs=gami_attrs).to_netcdf(output / "gami_native_2020_reduced.nc")

    fra = pd.read_csv(fra_dir / "regional_2010_2020.csv")
    frag = pd.read_csv(fra_dir / "global_2010_2020.csv")
    fra = pd.concat([fra[fra.year.eq(2020)], frag[frag.year.eq(2020)]], ignore_index=True)
    fra.to_csv(output / "fra2025_reported_2020_with_coverage.csv", index=False)
    comparisons = []
    for region in REGIONS + ["GLOBAL"]:
        m = model[model.region.eq(region)]
        g = gami[gami.region.eq(region)]
        farea = fra[fra.reg_land.eq(region) & fra.variable.eq("forest_area_Mha")].iloc[0]
        flive = fra[fra.reg_land.eq(region) & fra.variable.eq("living_carbon_PgC")].iloc[0]
        row = {"region": region, "fra_forest_area_Mha": float(farea.available_subtotal),
            "fra_live_carbon_available_PgC": float(flive.available_subtotal),
            "fra_live_carbon_complete_PgC": float(flive.complete_total),
            "fra_live_forest_area_coverage": float(flive.forest_area_coverage_fraction),
            "fra_carbon_missing_countries": int(flive.missing_country_count),
            "fra_covered_forest_density_MgC_ha": float(flive.available_subtotal / flive.covered_forest_area_Mha * 1000),
            "fra_regional_geography_strictly_equivalent": False,
            "fra_rowo_not_a_same_domain_comparison": region == "ROWO"}
        for field in ["forest_area_Mha", "live_carbon_PgC", "live_density_MgC_ha",
                      "known_young_share_total_forest_pct", "primary_unaged_share_total_forest_pct",
                      "known_young_share_aged_forest_pct"]:
            row.update({"model_" + field + "_" + stat: val for stat, val in describe(m[field]).items()})
        for field in ["forest_area_Mha", "young_share_total_forest_pct", "mature_code300_share_total_forest_pct"]:
            row.update({"gami_" + field + "_" + stat: val for stat, val in describe(g[field]).items()})
        row["model_live_to_fra_available_ratio_median"] = row["model_live_carbon_PgC_median"] / row["fra_live_carbon_available_PgC"]
        row["model_area_to_fra_ratio_mean"] = row["model_forest_area_Mha_mean"] / row["fra_forest_area_Mha"]
        comparisons.append(row)
    comp = pd.DataFrame(comparisons)
    comp.to_csv(output / "regional_display_statistics.csv", index=False)

    dictionary = pd.read_csv(dictionary_path).set_index("reg_code")["IAMC_R10"].to_dict()
    with xr.open_dataset(map_path) as ds:
        raw = ds.frac_reg.transpose("lat", "lon", "reg_code").values
        codes = ds.reg_code.values.astype(str)
        rf = np.zeros((11, ds.sizes["lat"], ds.sizes["lon"]), dtype=np.float32)
        for j, region in enumerate(REGIONS):
            selected = [k for k, c in enumerate(codes) if c != "XXX" and dictionary.get(c) == region]
            rf[j] = np.nansum(raw[:, :, selected], axis=2)
        covered = np.nansum(rf, axis=0)
        dominant = np.where(covered > 0, np.argmax(rf, axis=0), -1).astype(np.int16)
        grid = xr.Dataset({"dominant_region_index": (("lat", "lon"), dominant),
            "region_fraction": (("region", "lat", "lon"), rf)},
            coords={"region": REGIONS, "lat": ds.lat.values, "lon": ds.lon.values},
            attrs={"map_policy": "Existing one-degree OSCAR geography; dominant geographic region for regional-value choropleths only. No gridded forest simulation or inferred within-region forest distribution.",
                   "source_map_metadata": json.dumps(dict(ds.attrs))})
    grid.to_netcdf(output / "regional_map_geometry.nc", encoding={"region_fraction": {"zlib": True}})
    inputs = {"emissions_H1_checkpoint": str(state_path.relative_to(root)),
              "GAMI_prepared_native_support": str(obs_path.relative_to(root)),
              "FRA2025_prepared_inventory": str(fra_dir.relative_to(root)),
              "native_map_geometry": str(map_path.relative_to(root)),
              "native_region_dictionary": str(dictionary_path.relative_to(root))}
    summary = {"status": "complete_current_2020_observational_comparison", "reference_year": 2020,
        "model_run_identity": model_identity, "gami_reduction_identity": gami_identity,
        "model_original_members": 500, "gami_realizations": 20,
        "model_global": comp[comp.region.eq("GLOBAL")].iloc[0].to_dict(),
        "regional": comparisons,
        "gami_unassigned_forest_Mha": describe(gami[gami.region.eq("UNASSIGNED")].forest_area_Mha),
        "gami_unresolved_299_global_Mha": describe(gami[gami.region.eq("GLOBAL")].unresolved_code299_area_Mha)}
    manifest = {"status": summary["status"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "model_attrs": model_attrs, "gami_attrs": gami_attrs,
        "source_config": ids.tolist(), "source_config_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
        "model_time_support": "Boundary stock and area saved at model year 2020; observational products use nominal reference year 2020, not a matched acquisition instant.",
        "age_operator": "Model age = 2020 - real birth_time; forest_class sum; right-closed 20-year classes; primary unaged is distinct. GAMI native histogram uses original 0..300 codes: 299 unresolved, 300 mature terminal category, not exact age or model primary.",
        "forest_domains": "Each source retains its own forest support. Model: secdf+pltns+unaged primary. GAMI: mapped forest using ML and Landsat disturbance/height fusion, including some trees outside FAO forest and excluding temporarily unstocked areas. FRA: forest land, including qualifying young/temporarily unstocked forest.",
        "carbon_operator": "Model W+F live forest C including primary; FRA published AGC+BGC, no dry-mass conversion; both exclude soil/deadwood/litter/HWP. Organ and forest-domain definitions are approximate, not exact pointwise equivalence.",
        "missing_policy": "FRA available country subtotals and complete totals retained separately; no fill, upscale, or model rescaling. ROWO has a substantial known national-versus-grid geographic mismatch.",
        "uncertainty": "Model median and empirical 5th-95th percentiles over original joint parameters. GAMI summaries over 20 product realizations; not paired with model IDs and not observational confidence limits. No FRA uncertainty distribution is supplied.",
        "independence": "Neither this GAMI v2.1 map nor the FRA2025 snapshot was fitted in this workflow. Model age is driven by land-use history, not initialized from GAMI. GAMI depends on ESA-CCI AGB v5; shared plot/biomass or national-statistics ancestry with model growth data/host inputs is possible and not independently excluded. This is a consistency assessment, not independent validation of temperature or future age response.",
        "new_model_runs": False, "new_calibration": False}
    (output / "summary.json").write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(clean_json(manifest), indent=2, allow_nan=False) + "\n")
    print(comp[["region", "model_forest_area_Mha_mean", "gami_forest_area_Mha_mean",
                "model_known_young_share_total_forest_pct_median", "gami_young_share_total_forest_pct_median",
                "model_primary_unaged_share_total_forest_pct_median", "model_live_carbon_PgC_median",
                "fra_live_carbon_available_PgC"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())

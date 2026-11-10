#!/usr/bin/env python3
"""Summarize the four formal local experiments without pooling regions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import xarray as xr


def stats(values):
    a = np.asarray(values)
    return {"n": int(a.size), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "q05": float(np.quantile(a, .05)), "q95": float(np.quantile(a, .95)),
            "positive": int(np.sum(a > 0)), "negative": int(np.sum(a < 0)), "zero": int(np.sum(a == 0))}


def curves(da, comparison, metric):
    a = da.transpose("relative_year", "reg_land", "config")
    q = np.quantile(a.values, [.05, .5, .95], axis=2)
    rows = []
    for ti, year in enumerate(a.relative_year.values):
        for ri, region in enumerate(a.reg_land.values):
            values = a.values[ti, ri]
            rows.append({"comparison": comparison, "metric": metric, "relative_year": int(year),
                         "region": str(region), "mean": float(np.mean(values)),
                         "q05": q[0, ti, ri], "median": q[1, ti, ri], "q95": q[2, ti, ri],
                         "positive": int(np.sum(values > 0)), "negative": int(np.sum(values < 0)), "n": values.size})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source_dir / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["parameter_members"] != 500:
        raise ValueError("The formal complete original-500 local experiment is required")
    e = xr.load_dataset(args.source_dir / "endpoints_all500.nc")
    f = xr.load_dataset(args.source_dir / "fluxes_all500.nc")
    if not np.array_equal(e.config.values, np.arange(500)) or not np.array_equal(e.source_config, f.source_config):
        raise ValueError("Original member order differs")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {"source_manifest": manifest, "interval": "Each flux year y is the integral over relative (y-1,y]",
               "comparison": "Within each schedule/region, subtract on minus off for each original member before summary",
               "parameter_spread": "Empirical original joint-parameter 5th-95th percentiles, not real-world confidence or probability",
               "regions": {}, "global_aggregation": "None; all eleven regional columns retain separate identities"}
    selected_f = ("production", "forest_production", "nonforest_production", "net_uptake", "cumulative_net_uptake",
                  "cumulative_production", "direct_age_production", "fire", "cwd_respiration", "soil_respiration", "hwp_respiration")
    selected_e = ("total_carbon", "forest_live_W", "forest_live_F", "cwd", "soil", "hwp",
                  "forest_area", "forest_area_years", "mean_forest_age")
    for label in f.experiment.values:
        for name in selected_f:
            rows.extend(curves(f[name].sel(experiment=label), str(label), name))
        for name in selected_e:
            rows.extend(curves(e[name].sel(experiment=label), str(label), name))
    paired = {}
    for schedule, a, b in (("pulse", "L1", "L2"), ("distributed", "L3", "L4")):
        df = f.sel(experiment=a).drop_vars(["schedule", "age_production", "experiment"], errors="ignore") - f.sel(experiment=b).drop_vars(["schedule", "age_production", "experiment"], errors="ignore")
        de = e.sel(experiment=a).drop_vars(["schedule", "age_production", "experiment"], errors="ignore") - e.sel(experiment=b).drop_vars(["schedule", "age_production", "experiment"], errors="ignore")
        paired[schedule] = (de, df)
        for name in selected_f:
            rows.extend(curves(df[name], schedule + "_on_minus_off", name))
        for name in selected_e:
            rows.extend(curves(de[name], schedule + "_on_minus_off", name))
        for ri, region in enumerate(e.reg_land.values):
            dst = summary["regions"].setdefault(str(region), {})
            dst[schedule] = {}
            for year in (1, 5, 20, 50, 100):
                cell = {name: stats(df[name].sel(relative_year=year, reg_land=region).values) for name in selected_f}
                cell["total_carbon"] = stats(de.total_carbon.sel(relative_year=year, reg_land=region).values)
                cell["forest_area_ha"] = float(e.forest_area.sel(experiment=a, relative_year=year, reg_land=region).mean())
                cell["forest_area_years"] = float(e.forest_area_years.sel(experiment=a, relative_year=year, reg_land=region).mean())
                rate = df.net_uptake.sel(relative_year=year, reg_land=region).values
                accum = df.cumulative_net_uptake.sel(relative_year=year, reg_land=region).values
                cell["joint_positive_rate_negative_cumulative_n"] = int(np.sum((rate > 0) & (accum < 0)))
                dst[schedule][str(year)] = cell
    for name in ("production", "net_uptake", "cumulative_net_uptake", "cumulative_production"):
        interaction = paired["pulse"][1][name] - paired["distributed"][1][name]
        rows.extend(curves(interaction, "pulse_minus_distributed_age_interaction", name))
    pd.DataFrame(rows).to_csv(args.output / "curves_summary.csv", index=False)
    # Source data preserve every member at physically specified landmark ages.
    landmark = xr.merge([e[list(selected_e)].rename({n: "endpoint_" + n for n in selected_e}),
                         f[list(selected_f)]], join="inner").sel(relative_year=[1, 5, 20, 50, 100])
    landmark.to_dataframe().reset_index().to_csv(args.output / "landmark_members_all500.csv.gz", index=False, compression="gzip")
    paired_landmarks = []
    for schedule, (_, df) in paired.items():
        block = df[["production", "net_uptake", "cumulative_net_uptake", "cumulative_production"]].sel(relative_year=[1, 5, 20, 50, 100]).to_dataframe().reset_index()
        block["schedule"] = schedule
        paired_landmarks.append(block)
    pd.concat(paired_landmarks).to_csv(args.output / "paired_landmark_members_all500.csv.gz", index=False, compression="gzip")
    summary["status"] = "complete"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    shutil.copy2(__file__, args.output / "source_used.py")
    print(json.dumps({"status": "complete", "output": str(args.output), "original_members": 500, "regional_columns": 11}))


if __name__ == "__main__":
    main()

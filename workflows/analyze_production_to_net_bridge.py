"""Paired 77-year production and directly recorded loss changes, VL/LN.

Uses completed matched-source futures only. No stages, age allocation of net
uptake, inferred residual mechanism, resampling or additional simulations.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "outputs/age_module/birth_source_age_response/PLAN.json"
DEFAULT_SOURCE = ROOT / "outputs/age_module/birth_source_age_response/analysis/source_annual_four_arms_all500.nc"
DEFAULT_OUTPUT = ROOT / "outputs/age_module/main_figure_redesign/analysis/production_to_net_bridge"
BACKGROUNDS = ("VL", "LN")
SELECTORS = ("00", "10", "11")
YEARS = np.arange(2023, 2100)
LOSSES = {
    "rf_Ecwd": "Actual coarse-woody-debris respiration",
    "rf_Esoil": "Actual soil respiration",
    "rf_Ehwp": "Actual harvested-wood-product respiration",
    "rf_Efire": "Actual fire carbon loss",
    "rf_Ecrop": "Actual crop-harvest carbon loss",
    "rf_Egraz": "Actual grazing carbon loss",
}
CONTRASTS = {
    "11_minus_00": ("11", "00", "All age responses, relative to no future age response"),
    "11_minus_10": ("11", "10", "Post2022-born response conditional on existing-cohort response enabled"),
}


def describe(v):
    v = np.asarray(v, dtype=float)
    if not np.isfinite(v).all():
        raise RuntimeError("A required original member has missing/nonfinite data")
    q = np.quantile(v, [.05, .5, .95])
    return dict(n=len(v), mean=float(v.mean()), q05=float(q[0]), median=float(q[1]),
        q95=float(q[2]), negative=int((v < 0).sum()), zero=int((v == 0).sum()),
        positive=int((v > 0).sum()))


def arm_paths(plan, bg):
    base = next(b for b in plan["baselines"] if b["scenario"] == bg)
    paths = dict(base["paths"])
    paths["10"] = next(c["output"] for c in plan["cells"]
        if c["scenario"] == bg and c["selector"] == "10")
    return {s: paths[s] for s in SELECTORS}


def collect(plan, source, output):
    ids = plan["source_config"]
    with xr.open_dataset(source) as f:
        if f.attrs.get("plan_identity") != plan["plan_identity"]:
            raise RuntimeError("The thin source belongs to a different intervention family")
        if f.source_config.values.tolist() != ids or not np.array_equal(f.future_year.values, YEARS):
            raise RuntimeError("The thin source has a different population/time support")
        thin = f.sel(scenario=list(BACKGROUNDS), selector=list(SELECTORS))
        data = xr.Dataset(coords=dict(scenario=list(BACKGROUNDS), selector=list(SELECTORS),
            future_year=YEARS, config=np.arange(500), source_config=("config", ids)))
        for name in ("NPP_PgC_yr", "net_land_uptake_PgC_yr"):
            data[name] = thin[name].load()
            data[name].attrs["units"] = "PgC yr-1"
    records = []
    arrays = {name: np.empty((2, 3, 77, 500)) for name in LOSSES}
    for bi, bg in enumerate(BACKGROUNDS):
        for si, (selector, directory) in enumerate(arm_paths(plan, bg).items()):
            path = ROOT / directory
            report = json.loads((path / "report.json").read_text())
            if report.get("members") != 500 or not report.get("status", "").startswith("passed"):
                raise RuntimeError("A source arm is not formally complete")
            records.append(dict(scenario=bg, selector=selector, path=directory,
                run_identity=report["run_identity"]))
            with xr.open_dataset(path / "annual_summary.nc") as f:
                used = f.sel(year=slice(2023, 2099))
                if f.source_config.values.tolist() != ids or not np.array_equal(used.year.values, YEARS):
                    raise RuntimeError("Annual member identity or support differs from the common source")
                for name in LOSSES:
                    v = used[name]
                    if v.attrs.get("units") != "PgC yr-1":
                        raise RuntimeError("Incompatible physical units: " + name)
                    spatial = [d for d in v.dims if d not in ("year", "config")]
                    arrays[name][bi, si] = v.sum(spatial, skipna=False).transpose("year", "config").values
            print(json.dumps(dict(status="source_read", scenario=bg, selector=selector)), flush=True)
    for name, values in arrays.items():
        data[name] = (("scenario", "selector", "future_year", "config"), values)
        data[name].attrs.update(units="PgC yr-1", definition=LOSSES[name])
    data.attrs.update(schema="production-to-net-bridge-source-v1", plan_identity=plan["plan_identity"],
        history_run_identity=plan["history_run_identity"],
        time_support="77 native annual means,2023-2099; sum annual rates times1yr; exclude2022",
        spatial_support="All native land classes/biomes, excluding permafrost",
        loss_overlap="rf_Eresp=rf_Ecwd+rf_Esoil; rf_Eresp is deliberately not added again",
        net_definition="D_Fland-D_Eluc=rf_NBP-rf_NBP_pi; common rf_NBP_pi cancels in paired differences")
    data.to_netcdf(output / "source_annual_global_fluxes.nc", engine="netcdf4",
        encoding={n: dict(zlib=True, complevel=2) for n in data.data_vars})
    return data, records


def analyze(args):
    plan = json.loads(args.plan.read_text())
    if plan["scenarios"] != list(BACKGROUNDS) or len(plan["source_config"]) != 500:
        raise RuntimeError("This analysis is fixed to the existing matched VL/LN original500")
    args.output.mkdir(parents=True, exist_ok=True)
    data, records = collect(plan, args.source, args.output)
    ids = plan["source_config"]
    rows, components, summaries, arm_rows = [], [], {}, []
    cumulative = data.sum("future_year", skipna=False)
    for bg in BACKGROUNDS:
        for selector in SELECTORS:
            arm = cumulative.sel(scenario=bg, selector=selector)
            for i, sid in enumerate(ids):
                row = dict(scenario=bg, selector=selector, member_position=i, source_config=sid)
                row.update({name + "_2023_2099_PgC": float(arm[name].values[i]) for name in data.data_vars})
                arm_rows.append(row)
        for contrast, (a, b, meaning) in CONTRASTS.items():
            delta = cumulative.sel(scenario=bg, selector=a) - cumulative.sel(scenario=bg, selector=b)
            d_npp = delta.NPP_PgC_yr.values
            d_net = delta.net_land_uptake_PgC_yr.values
            actual_loss_delta = sum(delta[n].values for n in LOSSES)
            compensation = -actual_loss_delta
            block = dict(meaning=meaning, n=500,
                delta_cumulative_NPP_PgC=describe(d_npp),
                delta_cumulative_net_land_uptake_PgC=describe(d_net),
                identified_loss_compensation_PgC=describe(compensation),
                actual_loss_changes_PgC={n: describe(delta[n].values) for n in LOSSES},
                production_lower_and_net_lower=int(((d_npp < 0) & (d_net < 0)).sum()),
                production_lower_and_identified_losses_lower=int(((d_npp < 0) & (compensation > 0)).sum()),
                compensation_fraction_ratio_of_means=(float(compensation.mean() / -d_npp.mean())
                    if d_npp.mean() < 0 else None))
            summaries.setdefault(bg, {})[contrast] = block
            for metric, values in [("delta_cumulative_NPP", d_npp), ("delta_cumulative_net_land_uptake", d_net),
                                   ("identified_loss_compensation", compensation)] + [
                                       ("delta_" + n, delta[n].values) for n in LOSSES]:
                components.append(dict(scenario=bg, contrast=contrast, metric=metric,
                    units="PgC over2023-2099", **describe(values)))
            for i, sid in enumerate(ids):
                row = dict(scenario=bg, contrast=contrast, member_position=i, source_config=sid,
                    delta_cumulative_NPP_PgC=d_npp[i], delta_cumulative_net_land_uptake_PgC=d_net[i],
                    identified_loss_compensation_PgC=compensation[i])
                for n in LOSSES:
                    row["delta_" + n + "_PgC"] = delta[n].values[i]
                rows.append(row)
    pd.DataFrame(rows).to_csv(args.output / "paired_member_cumulative_fluxes_all500.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(args.output / "arm_member_cumulative_fluxes_all500.csv", index=False)
    pd.DataFrame(components).to_csv(args.output / "component_summary.csv", index=False)
    summary = dict(status="complete", schema="production-to-net-bridge-v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(), scenarios=list(BACKGROUNDS),
        member_count_per_condition=500, years_inclusive=[2023, 2099], annual_labels=77,
        source_plan_identity=plan["plan_identity"], history_run_identity=plan["history_run_identity"],
        source_config=ids, contrasts=summaries,
        interpretation="Actual stored loss-channel changes, not an NPP-minus-net inferred mechanism. No age allocation of soil/HWP/net uptake.",
        empirical_range="5th-95th percentiles of paired original parameter members; not observational confidence intervals")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata = dict(source_thin=str(args.source.relative_to(ROOT)), source_plan=str(args.plan.relative_to(ROOT)),
        arms=records, nonoverlapping_actual_loss_variables=LOSSES,
        omitted_redundant_total="rf_Eresp=rf_Ecwd+rf_Esoil",
        carbon_routing="Harvest conversion transfers to dead/product pools are not all immediate atmospheric losses; HWP input is not a second loss term.",
        definition_sources=["oscar/_core/_model/_modules/regional_forest_age_native.py:223-240",
                            "oscar/_core/_model/_modules/regional_forest_age.py:548-558",
                            "workflows/analyze_future_age_response.py:60-72"],
        boundaries="Actual NPP includes coupled environmental responses. Source intervention11-10 is conditional, not an additive age-labelled net-sink share. No stage subsets or dropped members.")
    (args.output / "source_definitions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "README.md").write_text("# Production to net-uptake bridge\n\nCompleted matched VL/LN futures, original500 per condition,77 native annual means2023-2099. The two fixed paired contrasts are11-00 (all response) and11-10 (future-born response with existing response enabled). Summaries report empirical parameter-member percentiles. Production/net uptake come from the existing four-arm thin source; six actual loss channels are globally aggregated from the same canonical annual outputs. Eresp is excluded because it is exactly the CWD+soil aggregate. Compensation is the negative change in these directly recorded losses, not a residual labelled as mechanism. Common preindustrial NBP cancels in paired net-uptake changes. Carbon moved into products/dead pools is not counted again as immediate atmospheric release. No ages are assigned a share of mixed soil/product net uptake; actual production changes include coupled feedback.\n")
    shutil.copy2(Path(__file__), args.output / "analysis_source_used.py")
    print(json.dumps(dict(status="complete", output=str(args.output.relative_to(ROOT)))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()

"""Conditional future-cohort W and F production effects; no model is run here.

The new intermediate arm keeps existing-cohort gW/gF and future-cohort gW,
and replaces only future-cohort gF by one. The reused 00, 10 and 11 retain
their original scientific identities. Absolute paired effects are primary;
no direct-NPP fraction is converted into a temperature fraction.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.analyze_future_age_response import read_arm, describe, plain, YEARS
from workflows.analyze_restoration_emissions import trajectory_endpoints
from workflows.analyze_birth_source_age_response import interval_columns

SCHEMA = "matched-cmip7-establishment-response-sensitivity-analysis-v1"
PLAN_SCHEMA = "matched-cmip7-establishment-response-sensitivity-v1"
MODES = ("00", "10", "new", "11")


def contrasts(values):
    return dict(N1_W=values["new"]-values["10"],
                F_cond=values["11"]-values["new"],
                N1=values["11"]-values["10"],
                total_new=values["new"]-values["00"],
                total_main=values["11"]-values["00"],
                existing=values["10"]-values["00"])


def analyze(args):
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("scenarios") != ["VL", "LN"]:
        raise RuntimeError("A prepared matched VL/LN future-F sensitivity plan is required")
    ids = plan["source_config"]
    with xr.open_dataset(ROOT/plan["history_directory"]/"annual_summary.nc") as history:
        if history.source_config.values.tolist() != ids:
            raise RuntimeError("Common historical member order changed")
        offset = history.D_Tg.sel(year=slice(1850, 1900)).mean("year").values
        anchor_t = history.D_Tg.sel(year=2022).values
        anchor_co2 = history.D_CO2.sel(year=2022).values
    rows, arm_rows, populations, records, all_arrays, summary_rows = [], [], {}, [], [], []
    for bg in plan["scenarios"]:
        base = next(b for b in plan["baselines"] if b["scenario"] == bg)
        paths, expected = dict(base["paths"]), dict(base["run_identities"])
        cell = next(c for c in plan["cells"] if c["scenario"] == bg)
        report = json.loads((ROOT/cell["output"]/"report.json").read_text())
        if (report.get("status") != "passed_establishment_response_sensitivity"
            or report.get("members") != 500 or report.get("plan_identity") != plan["plan_identity"]):
            raise RuntimeError("The new sensitivity arm is incomplete or mismatched: " + bg)
        paths["new"], expected["new"] = cell["output"], report["run_identity"]
        arrays = {m: read_arm(ROOT/paths[m], ids, anchor_t, anchor_co2, offset) for m in MODES}
        if any(arrays[m]["run_identity"] != expected[m] for m in MODES):
            raise RuntimeError("A reused or new arm changed scientific identity")
        endpoints = {m: [trajectory_endpoints(YEARS, arrays[m]["T"][:, i]) for i in range(500)]
                     for m in MODES}
        for i, sid in enumerate(ids):
            ep = {m: endpoints[m][i] for m in MODES}
            p00 = int(ep["00"].peak_year)
            index = p00 - 2022
            interior = 2022 < p00 < 2099
            followup = interior and p00 + 20 <= 2099
            row = dict(scenario=bg, member_position=i, source_config=sid, p00_year=p00,
                       reference00_interior_maximum=interior,
                       reference00_resolved_overshoot=ep["00"].duration_defined,
                       reference00_plus20_defined=followup,
                       baseline00_11_resolved_overshoot=ep["00"].duration_defined and ep["11"].duration_defined,
                       all_four_resolved_overshoot=all(e.duration_defined for e in ep.values()))
            names = ("T_at_p00_K", "window_maximum_K", "selection_remainder_K", "gap20_K",
                     "net_to_p00_PgC", "net_2023_2099_PgC", "CO2_at_p00_ppm", "T2099_K",
                     "B_77_Kyr", "N_77_years")
            metrics = {k: {} for k in names}
            for mode in MODES:
                a, e = arrays[mode], ep[mode]
                t = a["T"][:, i]
                arm = dict(scenario=bg, selector=mode, member_position=i, source_config=sid,
                    status=e.status, window_maximum_K=e.peak_K, window_maximum_year=e.peak_year,
                    resolved_overshoot_peak=e.duration_defined, p00_year=p00, T_at_p00_K=t[index],
                    T2099_K=t[-1], CO2_at_p00_ppm=a["CO2"][index, i],
                    net_to_p00_PgC=a["net"][YEARS[1:] <= p00, i].sum(),
                    net_2023_2099_PgC=a["net"][:, i].sum(),
                    reference00_plus20_defined=followup,
                    gap20_K=t[index+20]-t[index] if followup else np.nan,
                    B_77_Kyr=np.maximum(t[1:]-1.5, 0).sum(),
                    N_77_years=int((t[1:] > 1.5).sum()))
                arm["selection_remainder_K"] = (e.peak_K-ep["00"].peak_K)-(t[index]-arrays["00"]["T"][index, i])
                for name in ("return_year", "duration_years"):
                    interval_columns(arm, name, getattr(e, name))
                arm_rows.append(arm)
                for name in names:
                    metrics[name][mode] = arm[name]
                for name in ("status", "window_maximum_year", "resolved_overshoot_peak"):
                    row[mode+"_"+name] = arm[name]
            for metric, values in metrics.items():
                for label, value in contrasts(values).items():
                    row[label+"_"+metric] = value
            rows.append(row)
        frame = pd.DataFrame([r for r in rows if r["scenario"] == bg])
        columns = [k for k in frame if k.startswith(("N1_W_", "F_cond_", "N1_", "total_new_", "total_main_", "existing_"))]
        masks = {
            "all_original_500": np.ones(500, dtype=bool),
            "reference00_resolved_overshoot": frame.reference00_resolved_overshoot.to_numpy(),
            "reference00_plus20_defined": frame.reference00_plus20_defined.to_numpy(),
            "reference00_resolved_plus20": (frame.reference00_resolved_overshoot & frame.reference00_plus20_defined).to_numpy(),
            "baseline00_11_resolved_overshoot": frame.baseline00_11_resolved_overshoot.to_numpy(),
            "all_four_resolved_overshoot": frame.all_four_resolved_overshoot.to_numpy()}
        summaries = {}
        for population, use in masks.items():
            chosen = frame.loc[use]
            values = {key: describe(chosen[key]) for key in columns}
            summaries[population] = dict(n=int(use.sum()), metrics=values)
            for name, stats in values.items():
                summary_rows.append(dict(scenario=bg, population=population,
                    n_original_members=500, n_population=int(use.sum()), metric=name, **stats))
        populations[bg] = dict(original_members=500, populations=summaries,
            status_counts={m: dict(Counter(e.status for e in endpoints[m])) for m in MODES})
        records.append(dict(scenario=bg, paths=paths, run_identities=expected))
        all_arrays.append(arrays)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/"member_establishment_contrasts_all500.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(args.output/"arm_members_all500.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output/"condition_sensitivity_summary.csv", index=False)
    ds = xr.Dataset(coords=dict(scenario=plan["scenarios"], selector=list(MODES), year=YEARS,
        future_year=YEARS[1:], config=np.arange(500), source_config=("config", ids)))
    for target, name, dimension in (("T_1850_1900_K", "T", "year"), ("CO2_anomaly_ppm", "CO2", "year"),
        ("net_land_uptake_PgC_yr", "net", "future_year"), ("NPP_PgC_yr", "NPP", "future_year")):
        ds[target] = (("scenario", "selector", dimension, "config"),
            np.stack([np.stack([a[m][name] for m in MODES]) for a in all_arrays]))
    ds.attrs.update(schema=SCHEMA, plan_identity=plan["plan_identity"], history_run_identity=plan["history_run_identity"],
        time_support="Shared historical annual2022 anchor; annual means2023-2099; integrals exclude2022",
        temperature_reference="Common H1 member-wise1850-1900 mean",
        net_land_definition="D_Fland-D_Eluc; all native land classes/pools excluding permafrost")
    ds.to_netcdf(args.output/"source_annual_establishment_all500.nc", engine="netcdf4",
        encoding={k: dict(zlib=True, complevel=2) for k in ds.data_vars})
    result = dict(schema=SCHEMA, status="complete", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_identity=plan["plan_identity"], history_run_identity=plan["history_run_identity"],
        scenarios=plan["scenarios"], members_per_condition=500, new_scientific_arms=2,
        new_member_trajectories=1000, contrast_rows=len(rows), arm_rows=len(arm_rows),
        definitions=dict(N1_W="new-10: future W response when future F is unmodulated; existing response unchanged",
            F_cond="11-new: future F response conditional on original future W; existing response unchanged",
            N1="11-10: original future W/F response", total_new="new-00", total_main="11-00", existing="10-00",
            common_year="Each member's00 window-maximum year p00; actual peak support flagged separately",
            selection="Window-maximum contrast minus same-p00 T contrast; descriptive reselection, not an independent cause",
            gap20="Paired difference in signed T change from p00 to p00+20; original00 eligibility",
            net="Native2023 prefix to p00 or2099; PgC uptake; no2022 integration",
            fixed="Gross drivers, event rules and establishment timing; actual supply/worked area remains responsive"),
        sources=records, populations=populations,
        limits=dict(intervention="Conditional sequential effects, not independently validated NPP curves or complete structural uncertainty",
            fractions="Absolute per-ID effects; no direct-production share converted to climate share or forced100percent attribution",
            origin="Actual birth_time; not carbon-atom ancestry; primary unaged",
            radiative="No exact four-arm ERF attribution from annual data",
            support="All original500 retained; same reference supports, endpoints and censoring explicit",
            uncertainty="Within-condition empirical joint-parameter distributions; pathways not pooled"))
    (args.output/"summary.json").write_text(json.dumps(plain(result), indent=2, allow_nan=False)+"\n")
    shutil.copyfile(Path(__file__), args.output/"analysis_source_used.py")
    (args.output/"README.md").write_text("""# Future-cohort establishment response: a finite conditional intervention

Only future_F is changed to one; future_W and both existing-cohort responses keep their original evaluators. Completed00,10,11 are reused. Primary forest is unaged. The common H1 end2022 state and complete matched VL/LN drivers are preserved.

N1_W=new-10 and F_cond=11-new are computed per original ID on the same p00 calendar before window maxima. Their telescoping sum is N1=11-10 for the same metric/support, but this order-dependent decomposition does not identify a general W/F interaction or universal temperature shares. The new F setting does not instantaneously mature F stocks. All carbon, environment and supply feedbacks continue.

The member table has1000 paired rows; the four-arm table has4000 rows. Keep all500 and report reference peak/20-year eligibility separately. Return intervals are not imputed as2099. Complete net is D_Fland-D_Eluc across all native land classes/pools, excluding permafrost; it is not a W/F-labelled net sink. Main figures are unchanged; this limited analysis supports Supplementary interpretation.
""")
    print(json.dumps(dict(status="complete", contrast_rows=len(rows), arm_rows=len(arm_rows))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()

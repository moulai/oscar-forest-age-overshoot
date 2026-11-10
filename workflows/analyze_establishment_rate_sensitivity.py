"""Paired future-cohort k plus/minus marginal-SE system responses.

Only future W uses a conditional curve; all F and existing responses remain.
Report rate-versus-main and future-cohort contrasts separately, with the original
member-specific 00 reference clock. These conditions are not confidence bounds.
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

SCHEMA = "matched-cmip7-establishment-rate-sensitivity-analysis-v1"
PLAN_SCHEMA = "matched-cmip7-establishment-rate-sensitivity-v1"
MODES = ("00", "10", "minus_se", "plus_se", "11")


def contrasts(values):
    return dict(k_plus_vs_main=values["plus_se"]-values["11"],
                k_minus_vs_main=values["minus_se"]-values["11"],
                k_plus_future=values["plus_se"]-values["10"],
                k_minus_future=values["minus_se"]-values["10"],
                k_plus_total=values["plus_se"]-values["00"],
                k_minus_total=values["minus_se"]-values["00"],
                N1_main=values["11"]-values["10"],
                total_main=values["11"]-values["00"],
                existing=values["10"]-values["00"])


def analyze(args):
    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("scenarios") != ["VL", "LN"]:
        raise RuntimeError("A prepared matched VL/LN k-rate sensitivity plan is required")
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
        for cell in [c for c in plan["cells"] if c["scenario"] == bg]:
            report = json.loads((ROOT/cell["output"]/"report.json").read_text())
            if (report.get("status") != "passed_establishment_rate_sensitivity"
                or report.get("members") != 500 or report.get("plan_identity") != plan["plan_identity"]):
                raise RuntimeError("The k-rate arm is incomplete or mismatched: " + bg)
            paths[cell["selector"]], expected[cell["selector"]] = cell["output"], report["run_identity"]
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
                       all_five_resolved_overshoot=all(e.duration_defined for e in ep.values()))
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
        columns = [k for k in frame if k.startswith(("k_plus_", "k_minus_", "N1_main_", "total_main_", "existing_"))]
        masks = {
            "all_original_500": np.ones(500, dtype=bool),
            "reference00_resolved_overshoot": frame.reference00_resolved_overshoot.to_numpy(),
            "reference00_plus20_defined": frame.reference00_plus20_defined.to_numpy(),
            "reference00_resolved_plus20": (frame.reference00_resolved_overshoot & frame.reference00_plus20_defined).to_numpy(),
            "baseline00_11_resolved_overshoot": frame.baseline00_11_resolved_overshoot.to_numpy(),
            "all_five_resolved_overshoot": frame.all_five_resolved_overshoot.to_numpy()}
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
    pd.DataFrame(rows).to_csv(args.output/"member_rate_contrasts_all500.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(args.output/"arm_members_all500.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output/"condition_rate_summary.csv", index=False)
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
    ds.to_netcdf(args.output/"source_annual_rate_conditions_all500.nc", engine="netcdf4",
        encoding={k: dict(zlib=True, complevel=2) for k in ds.data_vars})
    result = dict(schema=SCHEMA, status="complete", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_identity=plan["plan_identity"], history_run_identity=plan["history_run_identity"],
        scenarios=plan["scenarios"], members_per_condition=500, new_scientific_arms=4,
        new_member_trajectories=2000, contrast_rows=len(rows), arm_rows=len(arm_rows),
        definitions=dict(k_plus_vs_main="k_plus-11: future-cohort k+reportedSE conditional response",
            k_minus_vs_main="k_minus-11: future-cohort k-reportedSE conditional response",
            k_plus_future="k_plus-10: future response conditional on k+reportedSE and original F",
            k_minus_future="k_minus-10: future response conditional on k-reportedSE and original F",
            k_plus_total="k_plus-00",k_minus_total="k_minus-00",
            N1_main="11-10: original future W/F response",total_main="11-00",existing="10-00",
            common_year="Each member's00 window-maximum year p00; actual peak support flagged separately",
            selection="Window-maximum contrast minus same-p00 T contrast; descriptive reselection, not an independent cause",
            gap20="Paired difference in signed T change from p00 to p00+20; original00 eligibility",
            net="Native2023 prefix to p00 or2099; PgC uptake; no2022 integration",
            fixed="Gross drivers, event rules and establishment timing; actual supply/worked area remains responsive"),
        sources=records, populations=populations,
        limits=dict(intervention="Coherent one-factor plus/minus reported marginal-SE conditions, not confidence limits, independently validated NPP curves or complete structural uncertainty",
            fractions="Absolute per-ID effects; no direct-production share converted to climate share or forced100percent attribution",
            origin="Actual birth_time; not carbon-atom ancestry; primary unaged",
            radiative="No exact four-arm ERF attribution from annual data",
            support="All original500 retained; same reference supports, endpoints and censoring explicit",
            uncertainty="Within-condition empirical joint-parameter distributions; pathways not pooled"))
    (args.output/"summary.json").write_text(json.dumps(plain(result), indent=2, allow_nan=False)+"\n")
    shutil.copyfile(Path(__file__), args.output/"analysis_source_used.py")
    (args.output/"README.md").write_text("""# Future-cohort establishment-rate conditions

Only post-2022 W multipliers use the prepared k+reportedSE or k-reportedSE curve. Every F multiplier and pre-existing W multiplier retain the main evaluator. Original alpha, lambda calibration, initial states, parameters, gross drivers, events and numerical solver remain. Realised supply/standing area respond to the coupled state.

The two coherent rate conditions are not independent parameter draws, confidence bounds or a global structural-uncertainty envelope. No monotone ordering of gW, net or temperature is imposed. k-plus/main and k-minus/main isolate the one-factor conditional change; k-plus/10 and k-minus/10 retain the complete future-cohort effect relative to future-production-unmodulated cohorts. No F-by-k factorial is run.

The member table has 1000 rows containing both rate conditions per original ID; the five-arm table has 5000 rows. Original 00 reference maximum/+20 eligibility is retained (VL494/LN491, with resolved-overshoot subset separate). Window maxima, endpoints and return censoring remain explicit. Complete net is D_Fland-D_Eluc over native2023..2099, excluding permafrost; 2022 is only the shared climate anchor. Main figures are unchanged; Supplementary material receives the finite conditions.
""")
    print(json.dumps(dict(status="complete", contrast_rows=len(rows), arm_rows=len(arm_rows))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()

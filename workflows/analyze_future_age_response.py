"""Paired trajectories for five full CMIP7 pathways at common H1-2022 state.

Read the existing H1-on and the five new off_from_H1_2022 arms. No model is run.
Each pathway retains its original 500 parameter members; conditions are not pooled.
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
from workflows.analyze_restoration_emissions import trajectory_endpoints, subtract_intervals, median_bounds

YEARS = np.arange(2022, 2100)
SCHEMA = "full-cmip7-common-h1-future-age-response-analysis-v1"


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return ("+Infinity" if value > 0 else "-Infinity") if np.isinf(value) else None if np.isnan(value) else float(value)
    return value


def describe(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0)
    q = np.quantile(a, [.05, .5, .95])
    return dict(n=len(a), mean=a.mean(), q05=q[0], median=q[1], q95=q[2],
                negative_count=(a < 0).sum(), zero_count=(a == 0).sum(), positive_count=(a > 0).sum())


def global_config(value):
    return value.sum([d for d in value.dims if d not in ("year", "config")], min_count=1).transpose("year", "config").values


def read_arm(path, ids, anchor_t, anchor_co2, offset):
    with xr.open_dataset(path / "annual_summary.nc") as source:
        if source.source_config.values.tolist() != ids:
            raise RuntimeError("Member identity/order differs: " + str(path))
        used = source.sel(year=slice(2023, 2099))
        if not np.array_equal(used.year.values, YEARS[1:]):
            raise RuntimeError("Incomplete native future support")
        t = np.vstack([anchor_t, used.D_Tg.transpose("year", "config").values]) - offset[None]
        co2 = np.vstack([anchor_co2, used.D_CO2.transpose("year", "config").values])
        net = global_config(used.D_Fland - used.D_Eluc)
        npp = global_config(used.rf_NPP)
        run_identity = source.attrs.get("run_identity")
    return dict(T=t, CO2=co2, net=net, NPP=npp, run_identity=run_identity)


def interval_columns(row, prefix, name, interval):
    for key in ("lower", "upper", "lower_strict", "upper_strict"):
        row[prefix + name + "_" + key] = None if interval is None else getattr(interval, key)


def analyze(args):
    plan = json.loads(args.plan.read_text())
    ids = plan["source_config"]
    with xr.open_dataset(ROOT / plan["history_directory"] / "annual_summary.nc") as history:
        if history.source_config.values.tolist() != ids:
            raise RuntimeError("History identity/order differs from the prepared plan")
        offset = history.D_Tg.sel(year=slice(1850, 1900)).mean("year").values
        anchor_t = history.D_Tg.sel(year=2022).values
        anchor_co2 = history.D_CO2.sel(year=2022).values
    rows, sources, all_arrays, background_summaries = [], [], [], {}
    for cell in plan["cells"]:
        scenario = cell["scenario"]
        on_path, off_path = ROOT / cell["comparison_on"], ROOT / cell["output"]
        off_report = json.loads((off_path / "report.json").read_text())
        if off_report.get("status") != "passed_future_age_response" or off_report.get("members") != 500:
            raise RuntimeError("The new common-H1 off arm is incomplete")
        arrays = {"on": read_arm(on_path, ids, anchor_t, anchor_co2, offset),
                  "off": read_arm(off_path, ids, anchor_t, anchor_co2, offset)}
        if arrays["on"]["run_identity"] != cell["comparison_on_identity"]:
            raise RuntimeError("Existing on identity changed")
        endpoints = {mode: [trajectory_endpoints(YEARS, arrays[mode]["T"][:, i]) for i in range(500)]
                     for mode in ("on", "off")}
        for i, sid in enumerate(ids):
            ep = {m: endpoints[m][i] for m in ("on", "off")}
            row = dict(scenario=scenario, member_position=i, source_config=sid,
                       both_resolved_overshoot_peaks=all(e.duration_defined for e in ep.values()))
            for mode in ("on", "off"):
                t = arrays[mode]["T"][:, i]
                e = ep[mode]
                row.update({mode + "_status": e.status, mode + "_peak_K": e.peak_K,
                    mode + "_peak_year": e.peak_year, mode + "_T2099_K": t[-1],
                    mode + "_B_77_Kyr": np.maximum(t[1:] - 1.5, 0).sum(),
                    mode + "_N_77_years": (t[1:] > 1.5).sum(),
                    mode + "_net_uptake_2023_2099_PgC": arrays[mode]["net"][:, i].sum()})
                for name in ("return_year", "duration_years"):
                    interval_columns(row, mode + "_", name, getattr(e, name))
                for first, last in [(2060, 2079), (2080, 2099)]:
                    use = (YEARS >= first) & (YEARS <= last)
                    x = YEARS[use] - YEARS[use].mean()
                    row[f"{mode}_slope_{first}_{last}_K_per_year"] = np.sum(x * t[use]) / np.sum(x*x)
                    row[f"{mode}_change_{first}_{last}_K"] = t[YEARS == last][0] - t[YEARS == first][0]
            for metric in ["peak_K", "peak_year", "T2099_K", "B_77_Kyr", "N_77_years", "net_uptake_2023_2099_PgC",
                           "slope_2060_2079_K_per_year", "slope_2080_2099_K_per_year", "change_2060_2079_K", "change_2080_2099_K"]:
                row["delta_" + metric] = row["on_" + metric] - row["off_" + metric]
            gap = arrays["on"]["T"][:, i] - arrays["off"]["T"][:, i]
            p = int(ep["off"].peak_year)
            row["off_maximum_interior"] = 2022 < p < 2099
            row["same_year_gap_at_off_max_K"] = gap[p - 2022]
            row["maximum_selection_remainder_K"] = row["delta_peak_K"] - row["same_year_gap_at_off_max_K"]
            row["off_peak_plus20_defined"] = row["off_maximum_interior"] and p + 20 <= 2099
            row["off_peak_plus20_gap_growth_K"] = gap[p + 20 - 2022] - gap[p - 2022] if row["off_peak_plus20_defined"] else np.nan
            row["common_return_phase_defined"] = False
            for name in ("phase_t0", "phase_t1", "phase_on_rate_K_per_year", "phase_off_rate_K_per_year", "phase_gap_growth_K"):
                row[name] = np.nan
            if row["both_resolved_overshoot_peaks"] and ep["off"].status == "returned":
                t0 = max(int(ep["on"].peak_year), int(ep["off"].peak_year))
                r_off = int(ep["off"].return_year.lower)
                t1 = min(r_off, int(ep["on"].return_year.lower)) if ep["on"].status == "returned" else r_off
                if t1 > t0:
                    row.update(common_return_phase_defined=True, phase_t0=t0, phase_t1=t1,
                               phase_gap_growth_K=gap[t1-2022] - gap[t0-2022])
                    for mode in ("on", "off"):
                        t = arrays[mode]["T"][:, i]
                        row[f"phase_{mode}_rate_K_per_year"] = (t[t1-2022]-t[t0-2022])/(t1-t0)
            rows.append(row)
        frame = pd.DataFrame([r for r in rows if r["scenario"] == scenario])
        metrics = [c for c in frame if c.startswith("delta_")] + ["same_year_gap_at_off_max_K", "maximum_selection_remainder_K", "off_peak_plus20_gap_growth_K", "phase_gap_growth_K", "phase_on_rate_K_per_year", "phase_off_rate_K_per_year"]
        summary = dict(original_members=500, on_status=dict(Counter(e.status for e in endpoints["on"])),
                       off_status=dict(Counter(e.status for e in endpoints["off"])), populations={})
        for population, use in [("all_original_500", np.ones(500, bool)),
                                ("both_resolved_overshoot_peaks", frame.both_resolved_overshoot_peaks.to_numpy()),
                                ("common_return_phase", frame.common_return_phase_defined.to_numpy())]:
            summary["populations"][population] = dict(n=int(use.sum()), metrics={m: describe(frame.loc[use, m]) for m in metrics})
        if not frame.empty:
            for name in ("return_year", "duration_years"):
                intervals = [subtract_intervals(getattr(a, name), getattr(b, name)) for a, b in zip(endpoints["on"], endpoints["off"]) if a.duration_defined and b.duration_defined]
                bounds = median_bounds(intervals)
                summary["paired_" + name] = dict(eligible_n=len(intervals), median_identification_bounds=None if bounds is None else bounds.as_dict())
        background_summaries[scenario] = summary
        all_arrays.append(arrays)
        sources.append(dict(scenario=scenario, comparison_on=cell["comparison_on"], off=cell["output"],
                            on_identity=arrays["on"]["run_identity"], off_identity=arrays["off"]["run_identity"]))
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output / "paired_members_all500.csv", index=False)
    coordinates = dict(scenario=plan["scenarios"], production=["on", "off"],
        year=YEARS, future_year=YEARS[1:], config=np.arange(500), source_config=("config", ids))
    dataset = xr.Dataset(coords=coordinates)
    for target, name, dimension in [("T_1850_1900_K", "T", "year"), ("CO2_anomaly_ppm", "CO2", "year"),
                                     ("net_land_uptake_PgC_yr", "net", "future_year"), ("NPP_PgC_yr", "NPP", "future_year")]:
        dataset[target] = (("scenario", "production", dimension, "config"),
                          np.stack([np.stack([a[m][name] for m in ("on", "off")]) for a in all_arrays]))
    dataset.attrs.update(schema=SCHEMA, family=plan["comparison_family"], plan_identity=plan["plan_identity"],
        time_support="Historical annual2022 anchor; actual native annual means2023-2099; future integrals exclude2022",
        temperature_reference="Common H1 historical member-wise1850-1900 mean", net_land_definition="D_Fland-D_Eluc; all native land classes/pools excluding permafrost")
    dataset.to_netcdf(args.output / "source_annual_paired_all500.nc", engine="netcdf4",
                      encoding={k: dict(zlib=True, complevel=2) for k in dataset.data_vars})
    result = dict(schema=SCHEMA, status="complete", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_identity=plan["plan_identity"], comparison_family=plan["comparison_family"],
        members_per_condition=500, scenarios=plan["scenarios"], history_run_identity=plan["history_run_identity"],
        populations=background_summaries, sources=sources,
        limits=dict(on_native_diagnostics="Existing on arms have no matched native ERF/direct-production outputs; exact paired radiative attribution requires output-only on replay",
            uncertainty="Empirical parameter distributions within each pathway; not confidence intervals or probabilities across pathways",
            peak="All-member window maxima retained; resolved interior overshoot populations reported separately",
            censoring="Unknown post2099 returns remain interval-censored; no substitution by2099",
            production="Age NPP groups are not age-resolved net uptake or temperature shares"))
    (args.output / "summary.json").write_text(json.dumps(plain(result), indent=2, allow_nan=False) + "\n")
    shutil.copyfile(Path(__file__), args.output / "analysis_source_used.py")
    (args.output / "README.md").write_text("# Future forest-age response at a common H1-2022 state\n\nFive complete CMIP7 pathways, each using its own matched drivers and original 500 parameter members. The existing H1-on future is compared with production-off continued from the same full H1 emissions-2022 state. This is separate from own-history H0/H1 and from controlled restoration backgrounds.\n\n`paired_members_all500.csv` retains all IDs, endpoint/censoring states, paired metrics, and conditional peak/return-phase flags. `source_annual_paired_all500.nc` retains all global annual trajectories. `summary.json` reports parameter-ensemble distributions per pathway and explicit support. No pathways or schedules are pooled as one uncertainty distribution.\n\nNative process diagnostics are present for the new off arms. Existing on annual means alone do not identify exact same-environment direct NPP or additive radiative temperature contributions. Those require an output-only replay of the relevant on arm.\n")
    print(json.dumps(dict(status="complete", scenarios=plan["scenarios"], paired_member_rows=len(rows))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()

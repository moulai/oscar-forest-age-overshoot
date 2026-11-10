#!/usr/bin/env python3
"""Climate, forest retention and service consequences of young-forest protection.

Read only complete, formally published 500-member cells. The comparison changes
actual land and wood services and is not an age-only or equal-service effect.
No model is run, no original member is selected, and no condition is pooled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from workflows.analyze_age_temperature_compensation import describe, outcomes, return_goal, total

ROOT = Path(__file__).resolve().parents[1]
FOREST = {"primf", "secdf", "pltns"}
LOWER = {"0_to_5": 0, "5_to_10": 5, "10_to_20": 10, "20_to_30": 20,
         "30_to_40": 30, "40_to_60": 40, "60_to_100": 60, "100_plus": 100}
Q_FIELDS = {
    "Q_requested_PgC": "rf_sec_agb_request", "Q_achieved_PgC": "rf_sec_agb_achieved",
    "Q_shortfall_PgC": "rf_sec_agb_shortfall", "standing_actual_area_Mha": "rf_sec_agb_area",
    "standing_removed_W_PgC": "rf_sec_agb_removed_W", "standing_removed_F_PgC": "rf_sec_agb_removed_F",
    "standing_HWP_input_PgC": "rf_sec_agb_HWP", "standing_CWD_input_PgC": "rf_sec_agb_CWD",
    "standing_soil_input_PgC": "rf_sec_agb_soil",
}


def source_ids(dataset, expected):
    ids = dataset.source_config.values.astype(int)
    if not np.array_equal(ids, expected):
        raise ValueError("The complete original member order must agree with the PLAN")


def global_annual(data):
    return total(data).transpose("year", "config").load()


def annual_products(directory, ids, future_years):
    with xr.open_dataset(directory / "annual_summary.nc") as d:
        source_ids(d, ids)
        if not np.array_equal(d.sel(year=future_years).year.values, future_years):
            raise ValueError("Native future labels 2023..2099 are required")
        x = {
            "raw_T_K": global_annual(d.D_Tg.sel(year=future_years)),
            "CO2_anomaly_ppm": global_annual(d.D_CO2.sel(year=future_years)),
            "net_land_PgC_yr": global_annual((d.D_Fland - d.D_Eluc).sel(year=future_years)),
            "NPP_PgC_yr": global_annual(d.rf_NPP.sel(year=future_years)),
        }
        metrics = {
            "cumulative_net_uptake_PgC": x["net_land_PgC_yr"].sum("year").values,
            "cumulative_NPP_PgC": x["NPP_PgC_yr"].sum("year").values,
        }
        for label, variable in Q_FIELDS.items():
            metrics[label] = global_annual(d[variable].sel(year=future_years)).sum("year").values
        attrs = dict(d.attrs)
    with xr.open_dataset(directory / "detailed_age_annual.nc") as d:
        source_ids(d, ids)
        groups = list(d.detailed_age_group.values.astype(str))
        forest_groups = [g for g in groups if g != "nonforest_unaged"]
        area = d.age_area_Mha.sum("reg_land").transpose("year", "config", "detailed_age_group").load()
        live = d.age_live_carbon_PgC.sum(["reg_land", "tissue"]).transpose("year", "config", "detailed_age_group").load()
        birth = d.true_birth_area_Mha.sum("reg_land").transpose("year", "config", "birth_source").load()
        forest = area.sel(detailed_age_group=forest_groups).sum("detailed_age_group")
        metrics.update({
            "forest_saved_area_2099_Mha": forest.sel(year=2099).values,
            "forest_area_years_2023_2099_Mha_yr": forest.sum("year").values,
            "forest_sampled_live_2099_PgC": live.sel(year=2099, detailed_age_group=forest_groups).sum("detailed_age_group").values,
            "primary_saved_area_2099_Mha": area.sel(year=2099, detailed_age_group="primary_unaged").values,
        })
        for label in birth.birth_source.values.astype(str):
            metrics[f"true_birth_{label}_Mha"] = birth.sel(birth_source=label).sum("year").values
        age_metadata = {"age_area": dict(d.age_area_Mha.attrs),
                        "age_live": dict(d.age_live_carbon_PgC.attrs), "dataset": dict(d.attrs)}
    x["forest_area_Mha"] = forest
    x["age_area_Mha"] = area
    x["true_birth_area_Mha"] = birth
    return metrics, x, attrs, age_metadata


def age_metrics(area, cutoff):
    young = [g for g in LOWER if LOWER[g] < cutoff]
    older = [g for g in LOWER if LOWER[g] >= cutoff]
    return {
        "age_resolved_below_cutoff_area_2099_Mha": area.sel(year=2099, detailed_age_group=young).sum("detailed_age_group").values,
        "age_resolved_atleast_cutoff_area_2099_Mha": area.sel(year=2099, detailed_age_group=older).sum("detailed_age_group").values,
        "age_resolved_atleast40_area_2099_Mha": area.sel(year=2099, detailed_age_group=[g for g in LOWER if LOWER[g] >= 40]).sum("detailed_age_group").values,
    }


def service_products(directory, ids, key):
    """Actual event accounting; gross event areas are not multiplied by dt."""
    records = []
    metrics = {}
    annual = {}
    with xr.open_dataset(directory / "detailed_age_annual.nc") as d:
        source_ids(d, ids)
        conv = d.secondary_conversion_area_Mha.sum("reg_land").load()
        footprint = d.standing_harvest_footprint_Mha.sum("reg_land").load()
        nf = d.nonforest_activity_area_Mha.sum("reg_land").load()
        nf_dest = d.nonforest_event_destination.values.astype(str)
        nf_source = d.nonforest_event_source.values.astype(str)
        nf_kind = d.nonforest_event_kind.values.astype(str)
        event_names = d.nonforest_event.values.astype(str)
        destinations = conv.destination_class.values.astype(str)
        for support in conv.activity_support.values.astype(str):
            metrics[f"secondary_conversion_{support}_Mha"] = conv.sel(activity_support=support).sum(["year", "destination_class"]).values
            metrics[f"standing_footprint_{support}_Mha"] = footprint.sel(activity_support=support).sum("year").values
        metrics["cancelled_secondary_conversion_to_nonforest_Mha"] = conv.sel(activity_support="cancelled", destination_class=[x for x in destinations if x not in FOREST]).sum(["year", "destination_class"]).values
        for support in nf.supply_accounting.values.astype(str):
            subset = nf.sel(supply_accounting=support)
            metrics[f"nonforest_gross_{support}_Mha"] = subset.sum(["year", "nonforest_event"]).values
            est = subset.isel(nonforest_event=np.flatnonzero(np.isin(nf_dest, list(FOREST))))
            metrics[f"nonforest_to_forest_{support}_Mha"] = est.sum(["year", "nonforest_event"]).values
            annual[f"nonforest_to_forest_{support}_Mha"] = est.sum("nonforest_event").transpose("year", "config")
        annual["cancelled_secondary_conversion_Mha"] = conv.sel(activity_support="cancelled").sum("destination_class").transpose("year", "config")
        annual["cancelled_standing_footprint_Mha"] = footprint.sel(activity_support="cancelled").transpose("year", "config")
        for index, event in enumerate(event_names):
            vals = nf.isel(nonforest_event=index).sum("year").transpose("config", "supply_accounting").values
            for i, sid in enumerate(ids):
                record = dict(**key, member_position=i, source_config=int(sid), event=str(event),
                              source_class=nf_source[index], destination_class=nf_dest[index], event_kind=nf_kind[index])
                record.update({str(s): float(vals[i, j]) for j, s in enumerate(nf.supply_accounting.values.astype(str))})
                records.append(record)
        for dest in destinations:
            vals = conv.sel(destination_class=dest).sum("year").transpose("config", "activity_support").values
            for i, sid in enumerate(ids):
                record = dict(**key, member_position=i, source_config=int(sid), event="secondary_conversion",
                              source_class="secdf", destination_class=str(dest), event_kind="conversion")
                record.update({str(s): float(vals[i, j]) for j, s in enumerate(conv.activity_support.values.astype(str))})
                records.append(record)
    return metrics, annual, records


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return value.item()
    return value


def run(args):
    plan = json.loads(args.plan.read_text())
    ids = np.asarray(plan["source_config"], dtype=int)
    if len(ids) != 500 or len(plan["cells"]) != 8:
        raise ValueError("Expected the complete VL/LN four-cutoff original ensemble")
    args.output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(args.baseline_source) as d:
        source_ids(d, ids)
        cached = d.sel(scenario=["VL", "LN"], selector=["00", "11"]).load()
        base = cached.sel(selector="11")
    years = base.year.values.astype(int)
    future = base.future_year.values.astype(int)
    if not np.array_equal(years, np.arange(2022, 2100)) or not np.array_equal(future, np.arange(2023, 2100)):
        raise ValueError("Retain annual 2022 historical anchor and 77 native future labels")
    baseline = {}
    ageoff = {}
    for bg in ("VL", "LN"):
        cell = next(c for c in plan["cells"] if c["scenario"] == bg)
        m, a, attrs, age_attrs = annual_products(ROOT / cell["baseline_on"], ids, future)
        t = base.T_1850_1900_K.sel(scenario=bg).transpose("year", "config").values
        m.update(outcomes(t, years))
        m["CO2_2099_anomaly_ppm"] = base.CO2_anomaly_ppm.sel(scenario=bg, year=2099).values
        offset = a["raw_T_K"].sel(year=2023).values - base.T_1850_1900_K.sel(scenario=bg, year=2023).values
        baseline[bg] = (m, a, offset, attrs, age_attrs)
        ageoff[bg] = outcomes(cached.T_1850_1900_K.sel(scenario=bg, selector="00").transpose("year", "config").values, years)
    rows, coverage, service_rows, annual_cells, source_records = [], [], [], [], []
    for cell in plan["cells"]:
        directory = ROOT / cell["output"]
        rf = directory / "report.json"
        report = json.loads(rf.read_text()) if rf.exists() else {}
        formal = report.get("status") == "passed_supply_limited_young_protection" and report.get("members") == 500
        key = {k: cell[k] for k in ("cell", "scenario", "cutoff_years")}
        coverage.append(dict(**key, formally_published=formal, report_status=report.get("status", "not_formally_published"), source=str(directory)))
        if not formal: continue
        if report.get("plan_identity") != plan["plan_identity"]:
            raise ValueError("Published policy and PLAN identities differ")
        bg = cell["scenario"]
        bm, ba, offset, battrs, bageattrs = baseline[bg]
        pm, pa, attrs, ageattrs = annual_products(directory, ids, future)
        bt = base.T_1850_1900_K.sel(scenario=bg).transpose("year", "config").values
        t = np.vstack([bt[0], pa["raw_T_K"].values - offset[None, :]])
        pm.update(outcomes(t, years))
        pm["CO2_2099_anomaly_ppm"] = pa["CO2_anomaly_ppm"].sel(year=2099).values
        pm.update(age_metrics(pa["age_area_Mha"], cell["cutoff_years"]))
        bref = dict(bm, **age_metrics(ba["age_area_Mha"], cell["cutoff_years"]))
        sm, sa, sr = service_products(directory, ids, key)
        service_rows.extend(sr)
        eligible, rgoal, rlabels = return_goal(pm, bref)
        off_eligible, off_goal, off_labels = return_goal(pm, ageoff[bg])
        for i, sid in enumerate(ids):
            record = dict(**key, member_position=i, source_config=int(sid),
                          reference_is_returner=bool(eligible[i]), reference_return_outcome=str(rlabels[i]),
                          reference_return_goal_met=rgoal[i],
                          ageoff_reference_is_returner=bool(off_eligible[i]), ageoff_reference_return_goal_met=off_goal[i],
                          ageoff_reference_return_outcome=str(off_labels[i]),
                          original_age_response_window_max_K=float(bref["window_max_K"][i]-ageoff[bg]["window_max_K"][i]),
                          remaining_window_max_K_vs_ageoff=float(pm["window_max_K"][i]-ageoff[bg]["window_max_K"][i]))
            for label, data in (("policy", pm), ("baseline_on", bref)):
                for metric, value in data.items():
                    record[label + "_" + metric] = safe(value[i])
            for metric in pm:
                if metric not in ("state", "resolved_overshoot_peak", "left_truncated_upcrossing", "window_max_year"):
                    record["effect_" + metric] = float(pm[metric][i] - bref[metric][i])
            record.update({"policy_" + metric: float(value[i]) for metric, value in sm.items()})
            rows.append(record)
        data = {
            "T_1850_1900_K": (("year", "config"), t),
            "baseline_T_1850_1900_K": (("year", "config"), bt),
            "auxiliary_ageoff_T_1850_1900_K": (("year", "config"), cached.T_1850_1900_K.sel(scenario=bg, selector="00").transpose("year", "config").values),
            "forest_area_Mha": pa["forest_area_Mha"].rename(year="future_year"),
            "baseline_forest_area_Mha": ba["forest_area_Mha"].rename(year="future_year"),
            "net_land_PgC_yr": pa["net_land_PgC_yr"].rename(year="future_year"),
            "baseline_net_land_PgC_yr": ba["net_land_PgC_yr"].rename(year="future_year"),
            "age_area_Mha": pa["age_area_Mha"].rename(year="future_year"),
            "baseline_age_area_Mha": ba["age_area_Mha"].rename(year="future_year"),
            "true_birth_area_Mha": pa["true_birth_area_Mha"].rename(year="future_year"),
            "baseline_true_birth_area_Mha": ba["true_birth_area_Mha"].rename(year="future_year"),
        }
        data.update({n: a.rename(year="future_year") for n, a in sa.items()})
        data = {name: value.reset_coords(drop=True) if isinstance(value, xr.DataArray) else value for name, value in data.items()}
        annual_cells.append(xr.Dataset(data, coords={"year": years, "future_year": future,
            "config": np.arange(500), "source_config": ("config", ids)}).expand_dims(cell=[cell["cell"]]))
        source_records.append(dict(**key, run_identity=report.get("run_identity"), baseline_run_identity=cell["baseline_run_identity"],
                                   annual_attributes=attrs, detailed_age_attributes=ageattrs,
                                   baseline_attributes=battrs, baseline_age_attributes=bageattrs))
    pd.DataFrame(coverage).to_csv(args.output / "formal_coverage.csv", index=False)
    member = pd.DataFrame(rows)
    summaries, transitions = [], []
    if not member.empty:
        member.to_csv(args.output / "member_retention_outcomes_all500.csv", index=False)
        for name, g in member.groupby("cell", sort=False):
            key = {k: g.iloc[0][k] for k in ("cell", "scenario", "cutoff_years")}
            numeric = [c for c in g if c.startswith("effect_") or c.startswith("policy_") and g[c].dtype.kind in "fiu"] + ["original_age_response_window_max_K", "remaining_window_max_K_vs_ageoff"]
            for metric in numeric:
                pop = "both_returned_by_2099" if metric == "effect_return_year" else ("policy_returned_by_2099" if metric == "policy_return_year" else "all_original_members")
                summaries.append(dict(**key, population=pop, metric=metric, **describe(g[metric])))
            off_rr = g[g.ageoff_reference_is_returner]
            summaries.append(dict(**key, population="auxiliary_fixed_ageoff_returners", metric="ageoff_reference_return_goal_met",
                n=len(off_rr), mean=float(off_rr.ageoff_reference_return_goal_met.mean()) if len(off_rr) else None))
            rr = g[g.reference_is_returner]
            summaries.append(dict(**key, population="fixed_baseline_on_returners", metric="reference_return_goal_met",
                n=len(rr), mean=float(rr.reference_return_goal_met.mean()) if len(rr) else None))
            for (bs, ps), gg in g.groupby(["baseline_on_state", "policy_state"], sort=False):
                transitions.append(dict(**key, baseline_on_state=bs, policy_state=ps, n=len(gg), fraction_original_members=len(gg)/500))
        pd.DataFrame(summaries).to_csv(args.output / "condition_retention_summary.csv", index=False)
        pd.DataFrame(transitions).to_csv(args.output / "return_state_transitions.csv", index=False)
        pd.DataFrame(service_rows).to_csv(args.output / "member_land_service_events_all500.csv.gz", index=False, compression="gzip")
        thin = xr.concat(annual_cells, dim="cell", join="exact")
        thin.attrs.update(plan_identity=plan["plan_identity"], comparison="Policy minus unchanged matched age-on baseline; actual land and wood services differ",
            temperature_reference="Common H1 PI display offset derived only from unchanged baseline-on matching annual2023; historical annual2022 anchor retained",
            stocks="Age live carbon is saved derivative-consumed substep stock, not an analytic annual stock integral; area and production use native dt-weighted support",
            events="Annual gross event totals, not multiplied by dt again; gross cumulative footprints are not unique preserved hectares")
        thin.to_netcdf(args.output / "source_annual_retention_all500.nc", engine="h5netcdf",
            encoding={v: {"zlib": True, "complevel": 4} for v in thin.data_vars})
    result = dict(status="complete" if all(x["formally_published"] for x in coverage) else "partial_formal_coverage",
        completed_at_utc=datetime.now(timezone.utc).isoformat(), plan_identity=plan["plan_identity"],
        formal_cells=sum(x["formally_published"] for x in coverage), expected_cells=8,
        original_members_per_condition=500, member_rows=len(rows), conditions=coverage, sources=source_records,
        scientific_comparison="Hard protection of age<10/20/30/40 secondary-forest proposals, no transfer to older forest; same-source nonforest requests limited by opening land supply; original fossil/net-industrial and other background drivers fixed",
        auxiliary_reference="Existing matched ageoff temperatures may show how much the policy offsets the original age-response departure; not a new pure-age interaction or equal-service causal share",
        land_service_units="Unmet nonforest gross event area is not food or agricultural yield loss; near-degenerate area ranges reflect fixed land inputs and do not represent uncertain land/age maps",
        meaning="A forest-conservation scenario with changed actual land/wood services and possibly reduced new forest establishment; not pure age-priority causality, equal services, no-afforestation comparison or global optimum",
        uncertainty="Pair same original ID within each condition first; empirical median/5–95 and direction fractions, not real-world confidence/probability; no cross-condition sample pooling",
        window="P and r retain 2022 historical annual anchor through2099; N and B integrate future native2023..2099 only; actual return undefined stays NaN",
        area="Saved annual mean forest area and cumulative area-years; cumulative cancelled/selected gross hectares are not unique protected land; old-age area differences do not track individual parcel survival",
        net="D_Fland minus D_Eluc across native land and tracked pools, excluding permafrost; NPP is not net uptake",
        wood_service="Q request/achieved/shortfall use the existing model AGB projection, not total removed W+F or instantaneous atmospheric emissions; standing footprint is repeated gross worked area, not deforested hectares")
    (args.output / "summary.json").write_text(json.dumps(safe(result), indent=2) + "\n")
    shutil.copy2(args.plan, args.output / "source_plan.json")
    shutil.copy2(__file__, args.output / "analysis_source_used.py")
    (args.output / "README.md").write_text("# Supply-limited young-forest protection\n\nOnly formally published complete original-member cells are analyzed. See summary.json for the exact scientific comparison, temporal/uncertainty definitions and actual coverage. This policy-versus-age-on comparison includes changes in actual land and wood services.\n\nReproduce from the project root:\n\n```bash\npython workflows/analyze_land_supply_limited_young_retention.py --plan outputs/age_module/land_supply_limited_young_retention/PLAN.json --output outputs/age_module/land_supply_limited_young_retention/analysis\n```\n\nmember_retention_outcomes_all500.csv retains all IDs, outcomes and paired effects. condition_retention_summary.csv separates conditions and populations. member_land_service_events_all500.csv.gz records actual/requested/cancelled or unmet gross activity by event. source_annual_retention_all500.nc preserves the plotted climate, forest age, births and principal land-service thin series. Infeasible or unpublished cells are coverage entries, never zero climate effects.\n")
    print(json.dumps({"status": result["status"], "formal_cells": result["formal_cells"], "member_rows": len(rows)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=ROOT / "outputs/age_module/land_supply_limited_young_retention/PLAN.json")
    p.add_argument("--baseline-source", type=Path, default=ROOT / "outputs/age_module/birth_source_age_response/analysis/source_annual_four_arms_all500.nc")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/age_module/land_supply_limited_young_retention/analysis")
    run(p.parse_args())


if __name__ == "__main__":
    main()

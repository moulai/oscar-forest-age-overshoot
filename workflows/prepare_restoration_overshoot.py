#!/usr/bin/env python3
"""Prepare the fixed restoration-timing experiment. This never runs a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/processed/regional_forest_age_inputs_v1"
BASE = ROOT / "outputs/01_prepared_data/cmip7_n2o_mass_corrected_H0_v1"
PACKAGE = ROOT / "outputs/01_prepared_data/restoration_overshoot_v2"
PLAN = ROOT / "outputs/age_module/restoration_emissions/PLAN.json"
HISTORY = ROOT / "outputs/age_module/historical_emissions/H1"
SCENARIOS = ("H", "HL", "M", "VL", "LN")
STARTS = (2030, 2035, 2040, 2045, 2050, 2055, 2060)
SPANS = (5, 10, 15, 20)
DONORS = ("c3ann", "c4ann", "c3per", "c4per", "c3nfx", "pastr", "range")
FRACTION = 1.0 / 8.0
ANCHOR, END = 2022, 2099
AREA_ATOL_MHA = 1e-8


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).relative_to(ROOT))


def schedule_name(start, span):
    return f"start_{start}_span_{span}"


def output_path(scenario, start, span, age_production):
    return ROOT / "outputs/age_module/restoration_emissions" / scenario / schedule_name(start, span) / age_production


def calendar_window_weights(years, start, span):
    """Calendar [start,start+span) maps to native end-year labels start+1..start+span."""
    require(start + span <= END and start > ANCHOR, "Planting window exceeds calendar support")
    years = np.asarray(years)
    return ((years > start) & (years <= start + span)).astype(np.float64) / float(span)


def state_areas(state, metadata):
    """All class areas in their native order, retaining every provided member."""
    classes = list(map(str, metadata.luh_class.values))
    regions = list(map(str, metadata.reg_land.values))
    size = int(state.sizes["config"])
    area = np.zeros((size, len(regions), len(classes)), dtype=np.float64)
    area[:, :, classes.index("primf")] = state.rf_A_primary.sel(reg_land=regions).transpose("config", "reg_land").values
    for name in map(str, state.forest_class.values):
        area[:, :, classes.index(name)] = state.rf_A_forest.sel(reg_land=regions, forest_class=name).sum("birth_time").transpose("config", "reg_land").values
    for name in map(str, state.nonforest_class.values):
        area[:, :, classes.index(name)] = state.rf_A_nonforest.sel(reg_land=regions, nonforest_class=name).transpose("config", "reg_land").values
    require(np.isfinite(area).all() and area.min() >= -AREA_ATOL_MHA, "Invalid restart class areas")
    return xr.DataArray(area, dims=("config", "reg_land", "luh_class"),
                        coords={"config": state.config, "source_config": state.source_config,
                                "reg_land": regions, "luh_class": classes},
                        attrs={"units": "Mha", "scope": "Exact class areas; no biome aggregation"})


def opening_for_design(metadata):
    status_path = HISTORY / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    final = HISTORY / "final_state.nc"
    ready = status.get("status") == "passed_historical" and final.is_file()
    if ready:
        path = final
        with xr.open_dataset(path) as state:
            area = state_areas(state, metadata)
        require(area.sizes["config"] == 500, "Completed H1 emissions restart is not 500 members")
    else:
        # Area-only planning support. This is never an atmosphere/carbon restart.
        path = ROOT / "outputs/age_module/historical_concentration/H1/final_state.nc"
        with xr.open_dataset(path) as state:
            area = state_areas(state.isel(config=slice(0, 1)), metadata)
    return area, {
        "ready": ready, "history_directory": relative(HISTORY),
        "history_run_identity": status.get("run_identity") if ready else None,
        "area_design_source": relative(path),
        "area_design_members": int(area.sizes["config"]),
        "scope": ("All 500 emissions restart members checked" if ready else
                  "Area-only design from one concentration-history member; emissions restart and all-500 supply audit remain mandatory before any run"),
    }


def load_official_inputs():
    """Use the unchanged prepared 2022-2099 VL inputs; no endpoint extension."""
    baseline = xr.load_dataset(BASE / "future_vl_forcing.nc")
    requests = xr.load_dataset(BASE / "future_vl_requests.nc")
    require(np.array_equal(baseline.year, np.arange(ANCHOR, END + 1)),
            "Expected the original annual 2022-2099 forcing")
    require(np.array_equal(requests.year, baseline.year), "Request years differ from baseline")
    backgrounds = []
    for scenario in SCENARIOS:
        with xr.open_dataset(BASE / f"future_{scenario.lower()}_forcing.nc") as old:
            require(np.array_equal(old.year, baseline.year), "Background years differ: " + scenario)
            backgrounds.append(old.D_Eff.load().expand_dims(scenario=[scenario]))
    fossil = xr.concat(backgrounds, dim="scenario").to_dataset(name="D_Eff")
    fossil.attrs.update(scope="Only D_Eff varies across controlled backgrounds. Every other driver is VL.",
                        definition="Non-land CO2 emissions in the supplied file, including its industrial/CDR convention")
    source_info = {
        "baseline_forcing": relative(BASE / "future_vl_forcing.nc"),
        "secondary_requests": relative(BASE / "future_vl_requests.nc"),
        "background_forcing_files": {s: relative(BASE / f"future_{s.lower()}_forcing.nc") for s in SCENARIOS},
        "first_native_year": ANCHOR, "last_native_year": END,
        "existing_basic_arrays_unchanged": True, "endpoint_extension": False,
        "extrapolation": False,
        "scope": "Existing corrected CMIP7 prepared inputs, native 2022 anchor and annual intervals through 2099. No official 2099-to-2100 interval is used.",
    }
    return baseline, requests, fossil, source_info


def event_description(metadata):
    return (list(map(str, metadata.event.values)), list(map(str, metadata.event_source_class.values)),
            list(map(str, metadata.event_target_class.values)), list(map(str, metadata.event_kind.values)))


def supply_arrays(event_area, initial, metadata):
    """Opening-stock source budgets; same-year receipts cannot supply outgoing events."""
    names, source, target, kind = event_description(metadata)
    classes = list(map(str, metadata.luh_class.values))
    values = event_area.sel(year=slice(ANCHOR + 1, END), event=names).transpose("year", "reg_land", "event").values
    delta, outgoing = [], []
    for c in classes:
        delta.append(values[:, :, [i for i, x in enumerate(target) if x == c]].sum(axis=2)
                     - values[:, :, [i for i, x in enumerate(source) if x == c]].sum(axis=2))
        indices = [i for i, (s, k) in enumerate(zip(source, kind)) if s == c and (k == "conversion" or c != "secdf")]
        outgoing.append(values[:, :, indices].sum(axis=2))
    delta, outgoing = np.stack(delta, axis=-1), np.stack(outgoing, axis=-1)
    prefix = np.concatenate([np.zeros_like(delta[:1]), np.cumsum(delta, axis=0)[:-1]])
    opening = initial.values[:, None] + prefix[None]
    return values, delta, outgoing, opening, opening - outgoing[None]


def audit_schedule(event_area, initial, metadata):
    _, delta, outgoing, opening, margin = supply_arrays(event_area, initial, metadata)
    endpoint = initial.values + delta.sum(axis=0)[None]
    at = np.unravel_index(np.argmin(margin), margin.shape)
    mi, ti, ri, ci = at
    years = event_area.sel(year=slice(ANCHOR + 1, END)).year.values
    record = {
        "members": int(initial.sizes["config"]), "minimum_supply_margin_Mha": float(margin.min()),
        "minimum_endpoint_area_Mha": float(endpoint.min()),
        "minimum_margin_cell": {"source_config": int(initial.source_config.values[mi]), "year": int(years[ti]),
                               "region": str(initial.reg_land.values[ri]), "source": str(initial.luh_class.values[ci])},
        "maximum_total_land_drift_Mha": float(np.max(abs(endpoint.sum(axis=-1) - initial.values.sum(axis=-1)))),
    }
    require(record["minimum_supply_margin_Mha"] >= -AREA_ATOL_MHA, "Source supply exceeded: " + str(record))
    require(record["minimum_endpoint_area_Mha"] >= -AREA_ATOL_MHA, "Negative endpoint area")
    return record, endpoint


def build_schedules(baseline, initial, metadata):
    names, source, target, kind = event_description(metadata)
    classes = list(map(str, metadata.luh_class.values))
    regions = list(map(str, metadata.reg_land.values))
    selected = [i for i, (s, t, k) in enumerate(zip(source, target, kind))
                if s in DONORS and t == "secdf" and k == "conversion"]
    require(len(selected) == len(DONORS), "Expected exactly seven donor-to-secdf conversions")
    events = [names[i] for i in selected]
    untouched = [name for name in names if name not in events]
    original = baseline.event_area.sel(event=events)
    total = original.sel(year=slice(ANCHOR + 1, END)).sum("year")
    years = np.asarray(baseline.year.values)
    _, _, _, _, base_margin = supply_arrays(baseline.event_area, initial, metadata)
    old = np.zeros(base_margin.shape[1:], dtype=np.float64)
    for i in selected:
        old[:, :, classes.index(source[i])] += baseline.event_area.isel(year=slice(1, None), event=i).values
    totals = old.sum(axis=0)
    pieces, records, common_bound = [], [], 1.0
    baseline_audit, baseline_endpoint = audit_schedule(baseline.event_area, initial, metadata)
    for start in STARTS:
        for span in SPANS:
            identifier = schedule_name(start, span)
            w = calendar_window_weights(years, start, span)
            revised = (1.0 - FRACTION) * original + FRACTION * total * xr.DataArray(w, dims="year", coords={"year": years})
            # Keep the structural anchor empty.
            require(float(abs(revised.isel(year=0)).sum()) == 0.0, "Schedule altered the 2022 anchor")
            area = baseline.event_area.copy(deep=True)
            area.loc[{"event": events}] = revised.transpose("year", "reg_land", "event")
            require(np.array_equal(area.sel(event=untouched), baseline.event_area.sel(event=untouched)), "An unrelated event changed")
            gross_error = float(abs(revised.sum("year") - original.sum("year")).max())
            require(gross_error <= AREA_ATOL_MHA, "Regional/source gross establishment total changed")
            audit, endpoint = audit_schedule(area, initial, metadata)
            endpoint_error = float(np.max(abs(endpoint - baseline_endpoint)))
            require(endpoint_error <= AREA_ATOL_MHA, "Endpoint class areas changed")
            h = totals[None] * w[1:, None, None] - old
            land_slope = -h.copy()
            land_slope[:, :, classes.index("secdf")] = h.sum(axis=-1)
            opening_slope = np.concatenate([np.zeros_like(land_slope[:1]), np.cumsum(land_slope, axis=0)[:-1]])
            slope = opening_slope - h
            bounds = np.divide(base_margin, -slope[None], out=np.full_like(base_margin, np.inf),
                               where=np.broadcast_to(slope[None] < 0, base_margin.shape))
            limit_at = np.unravel_index(np.argmin(bounds), bounds.shape)
            limit = float(min(1.0, bounds[limit_at]))
            common_bound = min(common_bound, limit)
            mi, ti, ri, ci = limit_at
            audit.update(schedule=identifier, start=start, span=span, calendar_interval=[start, start + span],
                         native_end_years=list(range(start + 1, start + span + 1)), f_max=limit,
                         f_max_binding={"source_config": int(initial.source_config.values[mi]), "year": int(years[ti + 1]),
                                        "region": regions[ri], "source": classes[ci]},
                         gross_total_max_abs_difference_Mha=gross_error,
                         endpoint_class_area_max_abs_difference_Mha=endpoint_error,
                         other_events_byte_equal=True, secondary_carbon_requests_unchanged=True)
            records.append(audit)
            pieces.append(revised.expand_dims(schedule=[identifier]))
    require(FRACTION <= common_bound, "The fixed one-eighth fraction is not jointly feasible; do not resize it")
    schedules = xr.concat(pieces, dim="schedule").to_dataset(name="establishment_flux")
    report = {
        "fraction": FRACTION, "common_f_max": common_bound,
        "eligible_gross_establishment_Mha": float(total.sum()),
        "retimed_gross_establishment_Mha": float(total.sum() * FRACTION),
        "all_gross_establishment_Mha": float(baseline.event_area.sel(event=[names[i] for i, (t, k) in enumerate(zip(target, kind)) if t == "secdf" and k == "conversion"]).sum()),
        "retimed_gross_by_region_Mha": {r: float(total.sel(reg_land=r).sum() * FRACTION) for r in regions},
        "retimed_gross_by_event_Mha": {e: float(total.sel(event=e).sum() * FRACTION) for e in events},
        "baseline": baseline_audit, "windows": records, "area_atol_Mha": AREA_ATOL_MHA,
        "area_rule": "Opening class area supplies outgoing events; secdf harvest area is recalculated by the unchanged Q/inventory kernel, so only secdf conversions are reserved here.",
        "interpretation": "Cumulative gross establishment activity, not unique parcels, additional forest area or permanent protection.",
    }
    return schedules, report


def prepare(output=PACKAGE):
    require(output == PACKAGE, "Use the fixed versioned restoration package path")
    require(not output.exists(), "Refusing to overwrite an existing restoration input package")
    require((PLAN.parent / "HOLD.md").is_file(), "Restoration HOLD must remain present during preparation")
    metadata = xr.load_dataset(INPUTS / "land_metadata.nc")
    baseline, requests, fossil, source_info = load_official_inputs()
    initial, dependency = opening_for_design(metadata)
    schedules, audit = build_schedules(baseline, initial, metadata)
    with xr.open_dataset(INPUTS / "primary_parameters.nc") as p:
        ids = p.source_config.values.tolist()
    require(len(ids) == 500 and len(set(ids)) == 500, "Original parameter IDs are not the ordered 500")
    if dependency["ready"]:
        require(initial.source_config.values.tolist() == ids, "Emissions restart member order differs")
    output.mkdir(parents=True)
    products = {"background_forcing.nc": baseline, "secondary_requests.nc": requests,
                "fossil_backgrounds.nc": fossil, "establishment_flux.nc": schedules,
                "opening_class_area.nc": initial.to_dataset(name="class_area")}
    files = {}
    for name, ds in products.items():
        ds.to_netcdf(output / name)
        files[name] = {"path": relative(output / name), "sha256": digest(output / name), "bytes": (output / name).stat().st_size}
    write_json(output / "land_supply_audit.json", audit)
    protocol = {
        "schema": "restoration-overshoot-inputs-v2", "status": "prepared_on_hold", "model_run": False,
        "years": [ANCHOR, END], "fraction": FRACTION, "donors": DONORS,
        "time_rule": "Start is a calendar boundary. [start,start+span) uses native end-year labels start+1..start+span and birth times start+0.5..start+span-0.5.",
        "land_rule": "For each region and eligible source, d_s=(1-f)*d_VL+f*W*w_s. Urban and every other event remain exactly VL. No cap, cancellation, replacement or extra establishment.",
        "atmosphere_rule": "Only D_Eff changes across H/HL/M/VL/LN controlled backgrounds; all other non-land inputs are VL. CO2 and temperature are solved; CH4/N2O concentrations remain prescribed VL.",
        "age_rule": "From one H1 emissions 2022 restart: on evaluates g(age); off sets W/F production multipliers to one. Both use real calendar ages and the same unchanged mortality/harvest/stock equations.",
        "dependency": dependency, "source": source_info, "files": files, "source_config": ids,
        "land_audit": relative(output / "land_supply_audit.json"),
        "scientific_question": "Can an earlier restoration start lower peak warming while lengthening the post-peak drawdown?",
        "not_claimed": "No absolute restoration benefit, no required-CDR inference, no full-CMIP7-pathway projection.",
    }
    write_json(output / "manifest.json", protocol)
    cells = []
    for scenario in SCENARIOS:
        for start in STARTS:
            for span in SPANS:
                for mode in ("on", "off"):
                    cells.append({"scenario": scenario, "start": start, "span": span, "age_production": mode,
                                  "schedule": schedule_name(start, span), "calendar_interval": [start, start + span],
                                  "native_end_years": list(range(start + 1, start + span + 1)),
                                  "birth_times": [y - 0.5 for y in range(start + 1, start + span + 1)],
                                  "output": relative(output_path(scenario, start, span, mode))})
    plan = {
        "schema": "restoration-overshoot-plan-v2", "status": "prepared_on_hold", "model_runs_started": 0,
        "cells": cells, "source_config": ids,
        "ordered_source_ids_sha256": "ae263948f561c541cb3f6fcec5d8f2cc399b51c108b4613be61c626b598df81e",
        "history_directory": relative(HISTORY), "history_run_identity": dependency["history_run_identity"],
        "restart_dependency": dependency, "prepared_inputs": relative(output),
        "input_manifest_sha256": digest(output / "manifest.json"), "years": [ANCHOR, END],
        "time_rule": protocol["time_rule"],
        "primary_schedules": {"early": schedule_name(2030, 5), "late": schedule_name(2050, 5)},
        "primary_backgrounds": ["VL", "LN"], "all_backgrounds_equal_scope": list(SCENARIOS),
        "land_fraction": FRACTION, "retimed_gross_establishment_Mha": audit["retimed_gross_establishment_Mha"],
        "common_f_max": audit["common_f_max"], "nt": 2, "dtype": "float64", "adapt_nt": False,
        "hold": relative(PLAN.parent / "HOLD.md"),
        "rule": protocol["land_rule"], "atmosphere_rule": protocol["atmosphere_rule"], "age_rule": protocol["age_rule"],
    }
    require(len(cells) == 280, "Unexpected formal matrix size")
    if dependency["ready"]:
        restart_record = {
            "status": "passed_data_audit", "model_run": False, "members": 500,
            "source_config": ids, "history_run_identity": dependency["history_run_identity"],
            "input_manifest_sha256": plan["input_manifest_sha256"], "windows": audit["windows"],
            "scope": "Actual H1 emissions 2022 class areas. All 28 source-supply budgets pass without caps.",
        }
        audit_path = PLAN.parent / "RESTART_SUPPLY_AUDIT.json"
        write_json(audit_path, restart_record)
        plan["restart_supply_audit"] = relative(audit_path)
        plan["restart_supply_audit_sha256"] = digest(audit_path)
    write_json(PLAN, plan)
    print(json.dumps({"status": plan["status"], "cells": len(cells), "end_year": END,
                      "retimed_gross_establishment_Mha": audit["retimed_gross_establishment_Mha"],
                      "common_f_max": audit["common_f_max"], "restart_ready": dependency["ready"],
                      "area_design_members": dependency["area_design_members"]}), flush=True)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    prepare()


if __name__ == "__main__":
    main()

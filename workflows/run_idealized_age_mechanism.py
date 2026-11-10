#!/usr/bin/env python3
"""Four local 1-ha age-mechanism experiments using the formal carbon kernel.

This is a local, fixed-environment mechanism experiment, not a set of global
climate simulations or an independent validation of the prescribed age curves.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
from oscar._core._model._modules.regional_forest_age import (
    STATE_NAMES, RegionalRates, advance_carbon, apply_annual_events,
)
from oscar._core._model._modules.regional_forest_age_reference import (
    fast_native_coefficients, cached_hwp_loss,
)
from oscar._core._model._modules.dynamic_reference import ReferenceState, advance_reference
from oscar._core._model._modules.dead_carbon import DeadCarbonRates

INPUTS = ROOT / "data/processed/regional_forest_age_inputs_v1"
HISTORY = ROOT / "outputs/age_module/historical_emissions/H1"
AREA_MHA = 1.e-6
T0 = 2022.5
CONFIGURATIONS = (("L1", "pulse", True), ("L2", "pulse", False),
                  ("L3", "distributed", True), ("L4", "distributed", False))
PGC_TO_MGC = 1.e9
FLUX_NAMES = ("production", "net_uptake", "fire", "crop_harvest", "grazing",
              "cwd_respiration", "soil_respiration", "hwp_respiration")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def carbon_by_region(state):
    return (state.rf_W_primary.values + state.rf_F_primary.values
            + state.rf_W_forest.values.sum(axis=(2, 3))
            + state.rf_F_forest.values.sum(axis=(2, 3))
            + state.rf_C_nonforest.values.sum(axis=2)
            + state.rf_C_cwd.values.sum(axis=2)
            + state.rf_C_soil.values.sum(axis=2)
            + state.rf_C_hwp.values.sum(axis=(2, 3)))


def prepare_batch(start, stop):
    member_slice = slice(start, stop)
    _, layout, kernels, _ = load_runtime_inputs(
        INPUTS, births=T0 + np.arange(20), tau_years=2,
        age_enabled=True, member_slice=member_slice)
    event = next(e for e in layout.events if e.event_id == "pastr_to_secdf")
    if (event.kind, event.vegetation_action, event.age_action, event.retained_split) != (
            "conversion", "retain", "reset", "source_p_wood"):
        raise ValueError("The formal pasture-establishment event semantics changed")
    layout = replace(layout, events=(event,))
    parameters = xr.load_dataset(HISTORY / "parameters.nc").isel(config=member_slice)
    environment = xr.load_dataset(INPUTS / "calibration_environment.nc").isel(config=member_slice)
    if not np.array_equal(parameters.source_config.values, layout.parameter_ids):
        raise ValueError("Historical and runtime original member identities differ")
    required = ["rf_A_nonforest", "rf_C_nonforest", "rf_C_cwd", "rf_C_soil", "rf_C_hwp",
                "rf_ref_live", "rf_ref_cwd", "rf_ref_soil"]
    with xr.open_dataset(HISTORY / "final_state.nc") as raw:
        source = raw[required].isel(config=member_slice).load()
    if not np.array_equal(source.source_config.values, parameters.source_config.values):
        raise ValueError("Common state member order differs")
    pasture_classes = [c.class_id for c in layout.classes if c.biome == "Pasture"]
    if pasture_classes != ["pastr"]:
        raise ValueError("This local experiment requires the documented pasture support")
    source_area = source.rf_A_nonforest.sel(nonforest_class="pastr").values
    eligibility = np.isfinite(source_area) & (source_area >= AREA_MHA)
    if not eligibility.all():
        bad = np.argwhere(~eligibility)
        raise ValueError("Unavailable 1-ha pasture donors, no substitution: " + repr(bad.tolist()))
    scale = AREA_MHA / source_area
    state = layout.empty_state(T0)
    pi = list(layout.biomes).index("Pasture")
    ni = list(layout.nonforest_classes).index("pastr")
    state.rf_A_nonforest.values[:, :, ni] = AREA_MHA
    state.rf_C_nonforest.values[:, :, ni] = source.rf_C_nonforest.sel(nonforest_class="pastr").values * scale
    for name in ("rf_C_cwd", "rf_C_soil"):
        state[name].values[:, :, pi] = source[name].sel(bio_land="Pasture").values * scale
    state.rf_C_hwp.values[:, :, pi, :] = source.rf_C_hwp.sel(bio_land="Pasture").values * scale[:, :, None]
    reference = ReferenceState(tuple(layout.parameter_ids), layout.regions, T0,
                               source.rf_ref_live.values.copy(), source.rf_ref_cwd.values.copy(),
                               source.rf_ref_soil.values.copy())
    layout.validate(state, T0)
    return parameters, environment, layout, kernels, state, reference, source_area


def make_event(state, layout, parameters, amount, relative_year):
    """Use the normal discrete event and its actual live/dead carbon routing."""
    demand = np.full((len(layout.regions), 1), amount, dtype=float)
    weights = np.ones((len(layout.regions), 1, len(layout.slots)))
    pwood = parameters.rf_p_wood.values
    phwp = parameters.rf_p_hwp.values
    pieces = []
    largest_residual = 0.
    for mi, member in enumerate(layout.members):
        single_layout = replace(layout, members=(member,), source_members=(layout.parameter_ids[mi],))
        single, ledger = apply_annual_events(
            state.isel(config=slice(mi, mi + 1)), single_layout, T0 + relative_year,
            demand, weights, pwood[mi:mi + 1], phwp[mi:mi + 1])
        largest_residual = max(largest_residual, float(abs(ledger.budget_residual).max()))
        pieces.append(single)
    return xr.concat(pieces, dim="config", data_vars="minimal", coords="minimal",
                     compat="equals", join="exact"), largest_residual


def frozen_environment_rates(parameters, environment, layout, reference, years):
    coefficients = fast_native_coefficients(environment, parameters, layout)
    product_loss = cached_hwp_loss(parameters, layout.empty_state(T0))
    prepared = []
    for _ in range(years * 2):
        step = advance_reference(reference, coefficients, .5)
        dead = tuple(DeadCarbonRates(layout.members, layout.biomes, layout.products,
                    step.cwd_rate[:, ri], step.soil_rate[:, ri],
                    coefficients.cwd_resp_fraction[:, ri], product_loss[:, ri])
                     for ri in range(len(layout.regions)))
        prepared.append(RegionalRates(coefficients.live_production, coefficients.live_mortality,
                        coefficients.live_fire, coefficients.live_crop, coefficients.live_graz,
                        coefficients.live_cwd_fraction, dead))
        reference = step.state
    return prepared, product_loss


def stock_fields(state, legacy_initial, product_loss, elapsed):
    legacy = legacy_initial * np.exp(-product_loss * elapsed)
    return {
        "forest_live_W": state.rf_W_forest.values.sum(axis=(2, 3)),
        "forest_live_F": state.rf_F_forest.values.sum(axis=(2, 3)),
        "nonforest_live": state.rf_C_nonforest.values.sum(axis=2),
        "cwd": state.rf_C_cwd.values.sum(axis=2),
        "soil": state.rf_C_soil.values.sum(axis=2),
        "hwp": state.rf_C_hwp.values.sum(axis=(2, 3)),
        "hwp_shared_legacy": legacy.sum(axis=(2, 3)),
        "hwp_new_event": (state.rf_C_hwp.values - legacy).sum(axis=(2, 3)),
        "total_carbon": carbon_by_region(state),
    }


def simulate_batch(task):
    start, stop, years, output = task
    tick = time.time()
    parameters, environment, layout, kernels, initial, reference, donor_area = prepare_batch(start, stop)
    rates, product_loss = frozen_environment_rates(parameters, environment, layout, reference, years)
    factors = []
    for step in range(years * 2):
        ages = np.maximum((step + .5) * .5 - np.arange(20), 0.)
        block = np.empty((stop - start, 11, 2, len(layout.forest_classes), 20))
        for ri, kernel in enumerate(kernels):
            block[:, ri] = kernel.factors(ages, member_ids=layout.parameter_ids)[:, :, None, :]
        factors.append(block)
    endpoints = []
    fluxes = []
    initial_total = carbon_by_region(initial)
    legacy_initial = initial.rf_C_hwp.values.copy()
    max_event_residual = 0.
    max_cumulative_residual = 0.
    for label, schedule, age_on in CONFIGURATIONS:
        state = initial.copy(deep=True)
        cumulative = np.zeros_like(initial_total)
        cumulative_forest_area = np.zeros_like(initial_total)
        end_data = {name: [] for name in stock_fields(state, legacy_initial, product_loss, 0)}
        end_data.update({"forest_area": [], "nonforest_area": [], "forest_area_years": [], "mean_forest_age": []})
        flow_data = {name: [] for name in FLUX_NAMES}
        flow_data.update({"forest_production": [], "nonforest_production": [], "direct_age_production": [],
                          "cumulative_net_uptake": [], "cumulative_production": []})
        cumulative_production = np.zeros_like(initial_total)
        for year in range(years + 1):
            if (schedule == "pulse" and year == 0) or (schedule == "distributed" and year < 20):
                amount = AREA_MHA if schedule == "pulse" else AREA_MHA / 20
                state, residual = make_event(state, layout, parameters, amount, year)
                max_event_residual = max(max_event_residual, residual)
            for name, value in stock_fields(state, legacy_initial, product_loss, year).items():
                end_data[name].append(value)
            area = state.rf_A_forest.values.sum(axis=(2, 3))
            end_data["forest_area"].append(area / AREA_MHA)
            end_data["nonforest_area"].append(state.rf_A_nonforest.values.sum(axis=2) / AREA_MHA)
            end_data["forest_area_years"].append(cumulative_forest_area.copy())
            age_area = (state.rf_A_forest.values * np.maximum(year - np.arange(20), 0.)[None, None, None, :]).sum(axis=(2, 3))
            end_data["mean_forest_age"].append(np.divide(age_area, area, out=np.full_like(area, np.nan), where=area > 0))
            if year == years:
                break
            annual = {name: np.zeros_like(initial_total) for name in flow_data}
            for half in range(2):
                si = year * 2 + half
                t = T0 + year + half * .5
                g = factors[si] if age_on else None
                if age_on:
                    annual["direct_age_production"] += np.sum(
                        rates[si].production[:, :, :2, None, None]
                        * state.rf_A_forest.values[:, :, None, :, :] * (g - 1.), axis=(2, 3, 4)) * .5
                cumulative_forest_area += state.rf_A_forest.values.sum(axis=(2, 3)) / AREA_MHA * .5
                state, ledger = advance_carbon(state, layout, t, t + .5, rates[si], g)
                for name in FLUX_NAMES:
                    annual[name] += ledger[name].values.sum(axis=2)
                annual["forest_production"] += (ledger.forest_wood_production.values + ledger.forest_other_production.values)
            annual["nonforest_production"] = annual["production"] - annual["forest_production"]
            cumulative += annual["net_uptake"]
            cumulative_production += annual["production"]
            annual["cumulative_net_uptake"] = cumulative.copy()
            annual["cumulative_production"] = cumulative_production.copy()
            residual = carbon_by_region(state) - initial_total - cumulative
            max_cumulative_residual = max(max_cumulative_residual, float(np.max(np.abs(residual))))
            for name, value in annual.items():
                flow_data[name].append(value)
        coords = {"relative_year": np.arange(years + 1), "config": list(layout.members),
                  "reg_land": list(layout.regions), "source_config": ("config", list(layout.parameter_ids))}
        e = xr.Dataset({name: (("relative_year", "config", "reg_land"), np.stack(value))
                        for name, value in end_data.items()}, coords=coords)
        for name in e:
            if name in ("forest_area", "nonforest_area"):
                e[name].attrs["units"] = "ha per initial ha"
            elif name == "forest_area_years":
                e[name].attrs["units"] = "ha yr per initial ha"
            elif name == "mean_forest_age":
                e[name].attrs["units"] = "yr"
            else:
                e[name] *= PGC_TO_MGC
                e[name].attrs["units"] = "MgC per initial ha"
        f = xr.Dataset({name: (("relative_year", "config", "reg_land"), np.stack(value))
                        for name, value in flow_data.items()}, coords={**coords, "relative_year": np.arange(1, years + 1)})
        for name in f:
            f[name] *= PGC_TO_MGC
            f[name].attrs["units"] = "MgC per initial ha" if name.startswith("cumulative") else "MgC per initial ha yr-1"
        endpoints.append(e.expand_dims(experiment=[label]))
        fluxes.append(f.expand_dims(experiment=[label]))
    endpoints = xr.concat(endpoints, dim="experiment", data_vars="all", coords="minimal", compat="override")
    fluxes = xr.concat(fluxes, dim="experiment", data_vars="all", coords="minimal", compat="override")
    for dataset in (endpoints, fluxes):
        dataset.coords["schedule"] = ("experiment", [v[1] for v in CONFIGURATIONS])
        dataset.coords["age_production"] = ("experiment", ["on" if v[2] else "off" for v in CONFIGURATIONS])
    endpoints["initial_pasture_source_area_Mha"] = (("config", "reg_land"), donor_area)
    batchdir = Path(output) / "batches"
    encoding = lambda dataset: {n: {"zlib": True, "complevel": 2} for n in dataset.data_vars}
    endpoints.to_netcdf(batchdir / f"endpoints_{start:03d}_{stop:03d}.nc", engine="h5netcdf", encoding=encoding(endpoints))
    fluxes.to_netcdf(batchdir / f"fluxes_{start:03d}_{stop:03d}.nc", engine="h5netcdf", encoding=encoding(fluxes))
    receipt = {"status": "complete", "member_start": start, "member_stop": stop,
               "source_config": list(map(int, layout.parameter_ids)), "regions": list(layout.regions),
               "years": years, "runtime_seconds": time.time() - tick,
               "max_event_carbon_residual_PgC": max_event_residual,
               "max_cumulative_carbon_residual_PgC": max_cumulative_residual,
               "donor_area_available": True}
    write_json(batchdir / f"receipt_{start:03d}_{stop:03d}.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/age_module/main_figure_redesign/idealized")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--interface-check", action="store_true", help="Only first original 10 members and one year")
    args = parser.parse_args()
    members, years = (10, 1) if args.interface_check else (500, 100)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "batches").mkdir(exist_ok=True)
    plan = {"status": "running", "kind": "interface_check" if args.interface_check else "formal_local_experiment",
            "configuration_count": 4, "regional_columns": 11, "parameter_members": members,
            "original_member_policy": "Unchanged IDs, order and joint parameters; no sampling or weighting",
            "climate_worlds": False, "relative_years": [0, years], "native_dt_years": .5,
            "area_ha": 1., "birth_times": (T0 + np.arange(20)).tolist(),
            "environment": str(INPUTS / "calibration_environment.nc"),
            "environment_support": "Fixed formal arithmetic 2000-2019 calibration environment, member and region specific",
            "common_state": str(HISTORY / "final_state.nc"), "parameters": str(HISTORY / "parameters.nc"),
            "source_event": "pastr_to_secdf: retain live biomass; reset birth; split by formal source p_wood",
            "initialization": "1 ha of common H1-2022 pasture; live/dead/product stocks scaled by actual Pasture area",
            "hwp_scope": "Shared biome-product legacy normalized by area, not ancestry of one physical hectare",
            "reference_scope": "Native density reference evolves under fixed E; soil input response remains active",
            "pulse": "1 ha at relative t0", "distributed": "1/20 ha at relative t0 through 19",
            "time_support": "Endpoint stocks after that integer-year event; flux label y integrates (y-1,y). Event has no instant atmospheric loss.",
            "interpretation": "Idealized internal mechanism; prescribed age-response curves, not independent validation",
            "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output / "manifest.json", plan)
    shutil.copy2(__file__, args.output / "source_used.py")
    tasks = [(a, min(a + args.batch, members), years, str(args.output)) for a in range(0, members, args.batch)]
    tick = time.time()
    receipts = []
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
        for future in as_completed([pool.submit(simulate_batch, task) for task in tasks]):
            receipt = future.result()
            receipts.append(receipt)
            print(json.dumps({"completed_original_members": sum(r["member_stop"] - r["member_start"] for r in receipts),
                              "target": members, "batch": [receipt["member_start"], receipt["member_stop"]],
                              "batch_seconds": round(receipt["runtime_seconds"], 1)}), flush=True)
    for stem in ("endpoints", "fluxes"):
        datasets = [xr.load_dataset(args.output / "batches" / f"{stem}_{a:03d}_{b:03d}.nc") for a, b, _, _ in tasks]
        combined = xr.concat(datasets, dim="config", data_vars="minimal", coords="minimal", compat="equals", join="exact")
        if not np.array_equal(combined.config.values, np.arange(members)):
            raise ValueError("Computational concatenation changed original member order")
        combined.attrs.update({k: str(v) for k, v in plan.items() if k not in ("status",)})
        combined.to_netcdf(args.output / f"{stem}_all500.nc", engine="h5netcdf",
                           encoding={n: {"zlib": True, "complevel": 3} for n in combined.data_vars})
    plan.update(status="complete", elapsed_seconds=time.time() - tick,
                complete_members=members, complete_regional_parameter_columns=11 * members,
                complete_configuration_columns=4 * 11 * members,
                max_event_carbon_residual_PgC=max(r["max_event_carbon_residual_PgC"] for r in receipts),
                max_cumulative_carbon_residual_PgC=max(r["max_cumulative_carbon_residual_PgC"] for r in receipts))
    write_json(args.output / "manifest.json", plan)
    # Combined formal products retain all original rows; computational shards
    # have no separate scientific identity and are removed after collection.
    shutil.rmtree(args.output / "batches")
    print(json.dumps({"status": "complete", "output": str(args.output), "seconds": round(time.time() - tick, 1)}), flush=True)


if __name__ == "__main__":
    main()

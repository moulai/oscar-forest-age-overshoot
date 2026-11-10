"""Analyze passive VL-early diagnostics along the two realized model paths.

Temperature contributions propagate paired effective-ERF differences from zero
with the original member-specific imex two-box update. They are realized-path
responses, not independent feedback interventions or age-specific warming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_age_peak_mechanism_diagnostics import COMPONENTS, GROUPS, SCHEMA, netcdf, write_json


def describe(values):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Undefined diagnostic was not assigned an explicit support")
    quantiles = np.quantile(x, [.05, .5, .95])
    return dict(n=len(x), mean=float(x.mean()), q05=float(quantiles[0]),
                median=float(quantiles[1]), q95=float(quantiles[2]),
                negative_count=int((x < 0).sum()), zero_count=int((x == 0).sum()),
                positive_count=int((x > 0).sum()))


def analyze(args):
    native = {}
    for mode in ("on", "off"):
        directory = args.diagnostics / mode
        path = directory / "native_mechanism.nc"
        if not path.exists():
            path = directory / "mechanism_native_steps.nc"
        native[mode] = xr.load_dataset(path)
    annual = {mode: xr.load_dataset(args.diagnostics / mode / "annual_summary.nc")
              for mode in ("on", "off")}
    production = {}
    for mode in ("on", "off"):
        path = args.diagnostics / mode / "direct_production.nc"
        if path.exists():
            production[mode] = xr.load_dataset(path)
        else:
            production[mode] = xr.Dataset({
                "direct_NPP_PgC_yr": annual[mode].mechanism_direct_NPP,
                "actual_NPP_PgC_yr": annual[mode].mechanism_actual_NPP_by_origin}).rename({
                    "mechanism_tissue": "tissue", "mechanism_forest_group": "forest_group",
                    "mechanism_origin": "origin", "mechanism_group_age_support": "age_support"})
    on, off = native["on"], native["off"]
    for name in ("config", "source_config", "left", "right", "forcing_component", "co2_component"):
        if not np.array_equal(on[name].values, off[name].values):
            raise ValueError("Paired identity/time/component mismatch: " + name)
    for name in on:
        if name.startswith("parameter_") or name.startswith("vLin_"):
            if not np.array_equal(on[name].values, off[name].values):
                raise ValueError("Pair changed climate or atmospheric parameters")
    if not all(np.array_equal(on[name][0], off[name][0]) for name in ("Tg", "Td", "CO2")):
        raise ValueError("Mechanism pair does not share the full 2022 climate/carbon state")
    identifiers = on.source_config.values
    years = annual["on"].year.values
    n, k = len(identifiers), len(COMPONENTS)
    g, d = np.zeros((n, k)), np.zeros((n, k))
    history_g, history_d = [], []
    delta_erf = (on.effective_ERF - off.effective_ERF).values
    p = {name: on["parameter_" + name].values[:, None]
         for name in ("Th_g", "Th_d", "lambda_0", "e_ohu", "th_0")}
    vg = on.vLin_D_Tg.values[:, None]
    vd = on.vLin_D_Td.values[:, None]
    for step in range(on.sizes["step"]):
        dt = float(on.right[step] - on.left[step])
        rate_g = (delta_erf[step] - p["lambda_0"] * g
                  - p["e_ohu"] * p["th_0"] * (g - d)) / p["Th_g"]
        rate_d = p["th_0"] * (g - d) / p["Th_d"]
        g, d = g + dt * rate_g / (1. + vg * dt), d + dt * rate_d / (1. + vd * dt)
        history_g.append(g.copy())
        history_d.append(d.copy())
    history_g, history_d = np.stack(history_g), np.stack(history_d)
    actual_g = np.concatenate([(on.Tg - off.Tg).values[1:],
                               (on.final_Tg - off.final_Tg).values[None]], axis=0)
    actual_d = np.concatenate([(on.Td - off.Td).values[1:],
                               (on.final_Td - off.final_Td).values[None]], axis=0)
    budget_signs = np.array([1., 1., 1., -1., -1., 1., 1.])
    budget_errors = {}
    for mode, ds in native.items():
        supplied = (ds.CO2_budget_input.values * budget_signs).sum(axis=-1) / ds.parameter_a_CO2.values
        budget_errors[mode] = float(np.abs(supplied - ds.d_CO2.values).max())
        if budget_errors[mode] > 1e-10:
            raise ValueError("Recorded CO2 budget inputs do not match the consumed derivative")
    error_g = float(np.abs(history_g.sum(axis=-1) - actual_g).max())
    error_d = float(np.abs(history_d.sum(axis=-1) - actual_d).max())
    if max(error_g, error_d) > 1e-10:
        raise ValueError("Recorded ERF stages do not reconstruct the realized imex temperature difference")
    annual_g = np.zeros((len(years), n, k))
    for j, year in enumerate(years):
        select = np.ceil(on.right.values).astype(int) == year
        weights = (on.right.values - on.left.values)[select]
        annual_g[j] = (history_g[select] * weights[:, None, None]).sum(axis=0) / weights.sum()
    annual_error = float(np.abs(annual_g.sum(axis=-1)
        - (annual["on"].D_Tg - annual["off"].D_Tg).values).max())
    if annual_error > 1e-10:
        raise ValueError("Native-stage decomposition does not match saved annual temperature support")

    reference = ROOT / "outputs/age_module/restoration_emissions/analysis/primary_VL/historical_temperature_reference.csv"
    refs = pd.read_csv(reference).set_index("source_config").loc[identifiers]
    anchor = refs.D_Tg_2022_raw_K.to_numpy()
    offset = refs.D_Tg_1850_1900_mean_K.to_numpy()
    all_years = np.r_[2022, years]
    temperature = {mode: np.vstack([anchor, annual[mode].D_Tg.values]) - offset
                   for mode in ("on", "off")}
    peaks = {mode: np.argmax(temperature[mode], axis=0) for mode in ("on", "off")}
    positions = np.arange(n)
    extended_g = np.concatenate([np.zeros((1, n, k)), annual_g], axis=0)
    at_off = extended_g[peaks["off"], positions]
    at_on = extended_g[peaks["on"], positions]
    peak_values = {mode: temperature[mode][peaks[mode], positions] for mode in ("on", "off")}
    same_year = (temperature["on"] - temperature["off"])[peaks["off"], positions]
    maximum_delta = peak_values["on"] - peak_values["off"]
    members = pd.DataFrame(dict(member_position=on.config.values, source_config=identifiers,
        on_window_max_year=all_years[peaks["on"]], off_window_max_year=all_years[peaks["off"]],
        on_window_max_K=peak_values["on"], off_window_max_K=peak_values["off"],
        delta_window_max_K=maximum_delta, delta_T_at_off_max_K=same_year,
        maximum_selection_remainder_K=maximum_delta - same_year))
    members["both_resolved_overshoot_peaks"] = (
        (members.on_window_max_year > 2022) & (members.on_window_max_year < all_years[-1])
        & (members.off_window_max_year > 2022) & (members.off_window_max_year < all_years[-1])
        & (members.on_window_max_K > 1.5) & (members.off_window_max_K > 1.5))
    for j, name in enumerate(COMPONENTS):
        members["response_at_off_max_" + name + "_K"] = at_off[:, j]
        members["response_at_on_max_" + name + "_K"] = at_on[:, j]
    prefix_mask = years[:, None] <= members.off_window_max_year.to_numpy()[None, :]
    prod_on, prod_off = production["on"], production["off"]
    for field, data in [("direct_on_path", prod_on.direct_NPP_PgC_yr),
                        ("observed_on_minus_off", prod_on.actual_NPP_PgC_yr - prod_off.actual_NPP_PgC_yr)]:
        values = data.sum("reg_land").transpose("year", "config", "tissue", "forest_group").values
        prefix = (values * prefix_mask[:, :, None, None]).sum(axis=0)
        for pool, tissue in enumerate(("W", "F")):
            for group, label in enumerate(GROUPS):
                members[f"prefix_{field}_{tissue}_{label}_PgC"] = prefix[:, pool, group]
    for tissue in ("W", "F"):
        columns = ["prefix_direct_on_path_" + tissue + "_" + group + "_PgC" for group in GROUPS]
        members["prefix_direct_on_path_" + tissue + "_PgC"] = members[columns].sum(axis=1)
    delta_budget = (on.CO2_budget_input - off.CO2_budget_input).values
    budget_mask = on.right.values[:, None] <= members.off_window_max_year.to_numpy()[None, :]
    dt = (on.right.values - on.left.values)[:, None, None]
    prefix_budget = (delta_budget * budget_signs[None, None, :] * dt * budget_mask[:, :, None]).sum(axis=0)
    for j, name in enumerate(on.co2_component.values):
        members["prefix_signed_atmospheric_input_" + str(name) + "_PgC"] = prefix_budget[:, j]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    members.to_csv(output / "member_mechanisms_all500.csv", index=False)
    times = xr.Dataset({
        "paired_effective_ERF_W_m2": (("step", "config", "forcing_component"), delta_erf),
        "paired_Tg_response_K": (("step", "config", "forcing_component"), history_g),
        "paired_Td_response_K": (("step", "config", "forcing_component"), history_d),
        "annual_paired_Tg_response_K": (("year", "config", "forcing_component"), annual_g),
        "paired_CO2_budget_input_PgC_yr": (("step", "config", "co2_component"), delta_budget)},
        coords={"step": on.step, "year": years, "config": on.config,
                "source_config": on.source_config, "forcing_component": list(COMPONENTS),
                "co2_component": on.co2_component, "left": on.left, "right": on.right})
    times.attrs.update(schema=SCHEMA, definition="Paired realized-path forcing responses from zero common-initial temperature difference",
        warning="Not independent feedback interventions, instantaneous forcing ratios or age-specific net/climate shares")
    netcdf(times, output / "paired_radiative_responses.nc")
    numeric = [c for c in members if c.startswith(("prefix_", "response_", "delta_", "maximum_"))]
    populations = {"all_original": np.ones(n, dtype=bool),
                   "both_resolved_overshoot_peaks": members.both_resolved_overshoot_peaks.to_numpy()}
    summary = dict(schema=SCHEMA, status="complete", scope="VL, 2030 start / five-year calendar span, production on minus off",
        member_count=n, original_member_count=500, source_config=identifiers.tolist(),
        climate_reconstruction_max_abs_K=dict(native_Tg=error_g, native_Td=error_d, annual_Tg=annual_error),
        consumed_CO2_budget_max_abs_ppm_yr=budget_errors,
        history_run_identity=on.attrs["history_run_identity"],
        input_manifest_sha256=on.attrs["input_manifest_sha256"],
        populations={name: dict(n=int(mask.sum()), metrics={c: describe(members.loc[mask, c]) for c in numeric})
                     for name, mask in populations.items() if mask.any()},
        interpretation="Actual-path radiative response; source production and instantaneous ERF fractions are not warming causal shares",
        source_locations=("Original formal run directories after diagnostic promotion" if args.published else str(args.diagnostics)),
        historical_reference=str(reference.relative_to(ROOT)), historical_reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())
    write_json(output / "summary.json", summary)
    import shutil
    shutil.copyfile(Path(__file__), output / "analysis_source_used.py")
    (output / "README.md").write_text("# Peak mechanism diagnostics\n\n"
        "The original 500 paired parameter members are retained. Direct NPP is evaluated at the same realized environment, area and state. "
        "Temperature components propagate actual paired effective ERF with the unchanged imex two-box update from zero initial differences. "
        "They describe realized-path radiative responses, not feedback-disabled causal effects.\n\n"
        "All-window maxima and the both-resolved-overshoot subset are distinct in the member table. The maximum-selection remainder is shown separately. "
        "Annual production prefixes use native intervals 2023 through each member's off maximum year. "
        "Atmospheric source prefixes end at that year's instantaneous endpoint, unlike annual-mean concentration.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published", action="store_true")
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()

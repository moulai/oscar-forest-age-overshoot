#!/usr/bin/env python3
"""Summarize all-member actual-age diagnostics across climate stages.

Climate stages follow the matched-future analysis. The direct production adjustment is
q*A*(g-1) along the realized on path; it is not a net-land or temperature share.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
AGE_GROUPS = ["0_to_5", "5_to_10", "10_to_20", "20_to_30", "30_to_40",
              "40_to_60", "60_to_100", "100_plus", "primary_unaged", "nonforest_unaged"]
FOREST_GROUPS = AGE_GROUPS[:9]


def quantiles(a):
    values = np.asarray(a)
    valid = np.isfinite(values)
    if not valid.any():
        return dict(n_defined=0, mean=np.nan, median=np.nan, q05=np.nan, q95=np.nan, positive_n=0, negative_n=0)
    b = values[valid]
    q = np.quantile(b, [.05, .5, .95])
    return dict(n_defined=int(valid.sum()), mean=float(b.mean()), q05=float(q[0]), median=float(q[1]),
                q95=float(q[2]), positive_n=int(np.sum(b > 0)), negative_n=int(np.sum(b < 0)))


def read_scenario(scenario, core):
    run = ROOT / "outputs/age_module/future_emissions" / scenario / "H1"
    path = run / "detailed_age_annual.nc"
    if not path.exists():
        raise FileNotFoundError("Detailed age diagnostics are unavailable: " + str(path))
    with xr.open_dataset(path) as ds:
        if tuple(ds.detailed_age_group.values) != tuple(AGE_GROUPS):
            raise ValueError("The actual detailed-age contract differs")
        if not np.array_equal(ds.config.values, np.arange(500)):
            raise ValueError("The original member axis changed")
        members = core[core.scenario == scenario].sort_values(["member_position", "stage_index"])
        expected_ids = members.drop_duplicates("member_position").source_config.values
        if not np.array_equal(ds.source_config.values, expected_ids):
            raise ValueError("Detailed-age and climate-stage member identities differ")
        source_attrs = dict(ds.attrs)
        fields = {}
        for name in ("age_area_Mha", "age_live_carbon_PgC", "age_actual_NPP_PgC_yr", "age_direct_NPP_PgC_yr"):
            v = ds[name]
            dims = [d for d in ("reg_land", "tissue") if d in v.dims]
            fields[name] = v.sum(dims).load()
        fields["true_birth_area_Mha"] = ds.true_birth_area_Mha.sum("reg_land").load()
        # Only actual clearance/standing carbon is called removed. Other
        # retained event transfers are not renamed as carbon destruction.
        processes = ["conversion_clear", "standing_secondary_harvest"]
        fields["event_source_age_area_Mha"] = ds.event_source_age_area_Mha.sel(event_process=processes).sum("reg_land").load()
        fields["event_source_age_carbon_PgC"] = ds.event_source_age_carbon_PgC.sel(event_process=processes).sum(["reg_land", "tissue"]).load()
    annual = xr.Dataset(fields)
    annual["forest_area_Mha"] = annual.age_area_Mha.sel(detailed_age_group=FOREST_GROUPS).sum("detailed_age_group")
    annual["young_0_to_5_area_Mha"] = annual.age_area_Mha.sel(detailed_age_group="0_to_5", drop=True)
    annual["forest_actual_NPP_PgC_yr"] = annual.age_actual_NPP_PgC_yr.sel(detailed_age_group=FOREST_GROUPS).sum("detailed_age_group")
    annual["forest_direct_NPP_PgC_yr"] = annual.age_direct_NPP_PgC_yr.sel(detailed_age_group=FOREST_GROUPS).sum("detailed_age_group")
    annual["forest_unmodulated_production_along_on_PgC_yr"] = annual.forest_actual_NPP_PgC_yr - annual.forest_direct_NPP_PgC_yr
    for name in annual.data_vars:
        annual[name].attrs["units"] = ("PgC yr-1" if "NPP" in name or "production_along" in name
                                       else "PgC" if "carbon" in name else "Mha")
    annual.attrs.update(source_attrs)
    return annual, members, path


def stage_values(annual, members):
    groups = annual.detailed_age_group.values
    stages = xr.Dataset(coords={"stage_index": np.arange(4), "config": annual.config,
                               "source_config": annual.source_config, "detailed_age_group": groups,
                               "birth_source": annual.birth_source, "event_process": annual.event_process})
    lengths = members.pivot(index="stage_index", columns="member_position", values="duration_years").reindex(index=range(4), columns=range(500)).values
    has = members.pivot(index="stage_index", columns="member_position", values="stage_has_intervals").reindex(index=range(4), columns=range(500)).values.astype(bool)
    complete = members.pivot(index="stage_index", columns="member_position", values="stage_complete").reindex(index=range(4), columns=range(500)).values.astype(bool)
    stages["duration_years"] = (("stage_index", "config"), lengths)
    stages["has_intervals"] = (("stage_index", "config"), has)
    stages["stage_complete"] = (("stage_index", "config"), complete)
    stages["common_four_nonempty_stages"] = ("config", has.all(axis=0))
    selected = ("age_area_Mha", "age_live_carbon_PgC", "age_actual_NPP_PgC_yr", "age_direct_NPP_PgC_yr",
                "true_birth_area_Mha", "event_source_age_area_Mha", "event_source_age_carbon_PgC")
    for field in selected:
        arr = annual[field].transpose("year", "config", ...)
        shape = (4, 500) + arr.shape[2:]
        mean = np.full(shape, np.nan)
        integral = np.full(shape, np.nan)
        for row in members.itertuples(index=False):
            if not row.stage_has_intervals:
                continue
            selected_years = ((arr.year.values >= row.first_annual_label) & (arr.year.values <= row.last_annual_label))
            value = arr.values[selected_years, int(row.member_position)]
            mean[int(row.stage_index), int(row.member_position)] = value.mean(axis=0)
            integral[int(row.stage_index), int(row.member_position)] = value.sum(axis=0)
        axes = ("stage_index", "config") + arr.dims[2:]
        stages[field + "_mean"] = (axes, mean)
        stages[field + "_integral"] = (axes, integral)
        annual_unit = annual[field].attrs["units"]
        stages[field + "_mean"].attrs["units"] = ("Mha yr-1" if field in ("true_birth_area_Mha", "event_source_age_area_Mha")
                                                      else "PgC yr-1" if field == "event_source_age_carbon_PgC" else annual_unit)
        stages[field + "_integral"].attrs["units"] = ("Mha" if field in ("true_birth_area_Mha", "event_source_age_area_Mha")
                                                          else "PgC" if "NPP" in field or field == "event_source_age_carbon_PgC"
                                                          else "Mha yr" if field == "age_area_Mha" else "PgC yr")
    stages.duration_years.attrs["units"] = "yr"
    return stages


def make_summary(annual, stages, scenario):
    rows = []
    # Two declared populations: each stage's available members, and one common
    # four-stage population so compositions can be read as one trajectory.
    common = stages.common_four_nonempty_stages.values.astype(bool)
    grouped = {
        "forest_area_Mha": stages.age_area_Mha_mean.sel(detailed_age_group=FOREST_GROUPS).sum("detailed_age_group", skipna=False),
        "young_0_to_5_area_Mha": stages.age_area_Mha_mean.sel(detailed_age_group="0_to_5"),
        "young_0_to_5_direct_NPP_PgC_yr": stages.age_direct_NPP_PgC_yr_mean.sel(detailed_age_group="0_to_5"),
        "forest_direct_NPP_PgC_yr": stages.age_direct_NPP_PgC_yr_mean.sel(detailed_age_group=FOREST_GROUPS).sum("detailed_age_group", skipna=False),
        "forest_actual_NPP_PgC_yr": stages.age_actual_NPP_PgC_yr_mean.sel(detailed_age_group=FOREST_GROUPS).sum("detailed_age_group", skipna=False),
        "true_birth_Mha_yr": stages.true_birth_area_Mha_mean.sum("birth_source", skipna=False),
    }
    grouped["young_0_to_5_area_fraction"] = grouped["young_0_to_5_area_Mha"] / grouped["forest_area_Mha"].where(grouped["forest_area_Mha"] > 0)
    for stage in range(4):
        for population in ("stage_available", "common_four_stages"):
            use = stages.has_intervals.sel(stage_index=stage).values.astype(bool)
            if population == "common_four_stages":
                use = use & common
            for name, arr in grouped.items():
                rows.append(dict(scenario=scenario, stage_index=stage, population=population, metric=name,
                                 n_population=int(use.sum()), **quantiles(arr.sel(stage_index=stage).values[use])))
            for field in ("age_area_Mha_mean", "age_actual_NPP_PgC_yr_mean", "age_direct_NPP_PgC_yr_mean"):
                for group in FOREST_GROUPS:
                    value = stages[field].sel(stage_index=stage, detailed_age_group=group).values[use]
                    rows.append(dict(scenario=scenario, stage_index=stage, population=population,
                                     metric=field + ":" + group, n_population=int(use.sum()), **quantiles(value)))
            for source in annual.birth_source.values:
                value = stages.true_birth_area_Mha_mean.sel(stage_index=stage, birth_source=source).values[use]
                rows.append(dict(scenario=scenario, stage_index=stage, population=population,
                                 metric="true_birth_Mha_yr:" + str(source), n_population=int(use.sum()), **quantiles(value)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenarios", nargs="+", choices=["H", "HL", "M", "VL", "LN"], default=["H", "HL", "M", "VL", "LN"])
    args = p.parse_args()
    members = pd.read_csv(args.core_analysis / "member_stage_carbon_all500.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    annuals, stage_sets, source_paths, rows = [], [], [], []
    for scenario in args.scenarios:
        annual, stage_members, path = read_scenario(scenario, members)
        stages = stage_values(annual, stage_members)
        rows.extend(make_summary(annual, stages, scenario))
        annuals.append(annual.expand_dims(scenario=[scenario]))
        stage_sets.append(stages.expand_dims(scenario=[scenario]))
        source_paths.append(str(path))
        print(json.dumps({"scenario": scenario, "members": 500,
                          "common_four_stage_n": int(stages.common_four_nonempty_stages.sum())}), flush=True)
    a = xr.concat(annuals, dim="scenario", data_vars="all", coords="minimal", compat="override")
    s = xr.concat(stage_sets, dim="scenario", data_vars="all", coords="minimal", compat="override")
    # A multi-path product must not inherit only the first path's run label.
    a.attrs = dict(scientific_family="Full matched CMIP7 emissions-driven H1 futures",
                   source_diagnostics_json=json.dumps(source_paths),
                   time_support="Native annual (y-1,y] means; actual gross events once at y-0.5",
                   direct_production="q*A*(g-1) along each realized age-on path, not full on-off or temperature allocation")
    s.attrs = dict(scientific_family=a.attrs["scientific_family"],
                   source_diagnostics_json=a.attrs["source_diagnostics_json"],
                   stages="Unmodified member-specific E0 on-defined u/p/r; all original stage flags retained")
    a.to_netcdf(args.output / "actual_age_annual_global_all500.nc", engine="h5netcdf",
                encoding={n: {"zlib": True, "complevel": 3} for n in a.data_vars})
    s.to_netcdf(args.output / "actual_age_stages_all500.nc", engine="h5netcdf",
                encoding={n: {"zlib": True, "complevel": 3} for n in s.data_vars})
    pd.DataFrame(rows).to_csv(args.output / "actual_age_stage_summary.csv", index=False)
    stage_members = members[members.scenario.isin(args.scenarios)].copy()
    common = stage_members.groupby(["scenario", "member_position"]).stage_has_intervals.all()
    stage_members["common_four_nonempty_stages"] = [bool(common.loc[(r.scenario, r.member_position)]) for r in stage_members.itertuples()]
    stage_members.to_csv(args.output / "stage_eligibility_all500.csv", index=False)
    manifest = dict(status="complete", scenarios=args.scenarios, original_members_per_condition=500,
                    source_diagnostics=source_paths, core_stage_source=str(args.core_analysis),
                    scientific_family="Five full matched CMIP7 emissions-driven H1 futures; not controlled restoration D_Eff-only backgrounds",
                    stage_definition="Unmodified E0 on-defined u/p/r annual intervals; off is not used to define ages",
                    state_support="Duration-weighted actual-step opening, post-event stocks; area is constant within each continuous step",
                    flux_support="Actual native production rates time-weighted into annual means; gross events once per actual annual routing",
                    age_groups=AGE_GROUPS, age_intervals="[0,5),[5,10),[10,20),[20,30),[30,40),[40,60),[60,100),[100,infinity); primary unaged",
                    direct_production="q*A*(g-1) along realized on state/environment; not full on-off, net uptake, CO2 or temperature allocation",
                    true_birth="Actual nonforest establishment/primary harvest/other new cohorts; standing secondary cuts retain birth",
                    inactive_relabel_boundary="Current pathways have no active plantation/keep-birth relabel events; no standing cut is counted as a birth",
                    populations="All 500 IDs and each stage status retained; complete nonempty four-stage population is explicitly flagged",
                    parameter_spread="Within each condition/population, empirical original joint-parameter 5th–95th percentiles; not real-world CI/probability")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(__file__, args.output / "source_used.py")
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()

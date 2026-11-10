#!/usr/bin/env python3
"""Connect completed birth-source interventions to the matched net-uptake gap.

Only full matched VL/LN outputs and their climate-stage calendars
are read. No model, new threshold, window or age-specific net attribution is used.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import xarray as xr

CONTRASTS={"total_on_minus_off":("11","00"),
           "future_response_existing_on":("11","10"),
           "future_response_existing_off":("01","00"),
           "remaining_existing_response":("10","00")}
MOSAIC_GROUPS={"under_20":["0_to_5","5_to_10","10_to_20"],
               "20_to_40":["20_to_30","30_to_40"],
               "40_to_100":["40_to_60","60_to_100"],
               "100_plus":["100_plus"],"primary_unaged":["primary_unaged"]}


def describe(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    if not a.size:return dict(n=0,mean=None,median=None,q05=None,q95=None,positive_n=0,negative_n=0)
    q=np.quantile(a,[.05,.5,.95]);return dict(n=int(a.size),mean=float(a.mean()),median=float(q[1]),q05=float(q[0]),q95=float(q[2]),positive_n=int((a>0).sum()),negative_n=int((a<0).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--birth-source",type=Path,required=True)
    p.add_argument("--core-analysis",type=Path,required=True)
    p.add_argument("--detailed-age-source",type=Path,
                   help="Global detailed-age annual cache for area-times-intensity production geometry")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with xr.open_dataset(args.birth_source) as src:d=src.load()
    if d.sizes["config"]!=500 or list(d.scenario.values)!=["VL","LN"]:raise ValueError("Original complete matched source is required")
    years=d.future_year.values.astype(int)
    if not np.array_equal(years,np.arange(2023,2100)):raise ValueError("Native 77-year support is required")
    ids=d.source_config.values.astype(int)
    outcomes=pd.read_csv(args.core_analysis/"member_outcomes_all500.csv")
    stages=pd.read_csv(args.core_analysis/"member_stage_carbon_all500.csv")
    stats=[];end_rows=[];stage_rows=[];datasets=[]
    for bg in d.scenario.values:
        obs=outcomes[outcomes.scenario==bg].sort_values("member_position")
        if not np.array_equal(obs.source_config.values,ids):raise ValueError("Climate-outcome source IDs differ")
        endpoint_index=obs.on_window_max_year.values.astype(int)-2023
        armd=d.sel(scenario=bg)
        for name,(a,b) in CONTRASTS.items():
            net=(armd.net_land_uptake_PgC_yr.sel(selector=a)-armd.net_land_uptake_PgC_yr.sel(selector=b)).transpose("future_year","config").values
            npp=(armd.NPP_PgC_yr.sel(selector=a)-armd.NPP_PgC_yr.sel(selector=b)).transpose("future_year","config").values
            metrics=dict(net_PgC_yr=net,NPP_PgC_yr=npp,NPP_minus_net_remainder_PgC_yr=npp-net)
            metrics.update({"cumulative_"+k.replace("_yr",""):np.cumsum(v,axis=0) for k,v in list(metrics.items())})
            datasets.append(xr.Dataset({k:(("scenario","contrast","future_year","config"),v[None,None]) for k,v in metrics.items()},
                coords=dict(scenario=[str(bg)],contrast=[name],future_year=years,config=d.config.values,source_config=("config",ids))))
            for metric,values in metrics.items():
                for j,y in enumerate(years):stats.append(dict(scenario=str(bg),contrast=name,metric=metric,year=int(y),**describe(values[j])))
            for i,sid in enumerate(ids):
                k=endpoint_index[i];rec=dict(scenario=str(bg),contrast=name,member_position=i,source_config=int(sid),
                    on_window_max_year=int(obs.iloc[i].on_window_max_year),on_resolved_overshoot_peak=bool(obs.iloc[i].on_resolved_overshoot_peak))
                for metric,values in metrics.items():
                    if metric.startswith("cumulative_"):
                        rec[metric+"_to_on_max"]=float(values[k,i]) if k>=0 else 0.
                        rec[metric+"_to_2099"]=float(values[-1,i])
                end_rows.append(rec)
            for st in stages[stages.scenario==bg].itertuples(index=False):
                i=int(st.member_position)
                if int(st.source_config)!=int(ids[i]):raise ValueError("Climate-stage source IDs differ")
                good=bool(st.stage_has_intervals)
                rec=dict(scenario=str(bg),contrast=name,member_position=i,source_config=int(ids[i]),
                    stage_index=int(st.stage_index),stage=str(st.stage),stage_support=str(st.stage_support),
                    has_intervals=good,first_annual_label=st.first_annual_label,last_annual_label=st.last_annual_label,
                    duration_years=st.duration_years,stage_complete=bool(st.stage_complete))
                for metric,values in list(metrics.items())[:3]:
                    use=values[int(st.first_annual_label)-2023:int(st.last_annual_label)-2022,i] if good else np.array([])
                    rec[metric+"_stage_mean"]=float(use.mean()) if good else np.nan
                    rec[metric.replace("_yr","")+"_stage_integral"]=float(use.sum()) if good else np.nan
                stage_rows.append(rec)
    result=xr.combine_by_coords(datasets,combine_attrs="drop_conflicts")
    result.attrs.update(scientific_family="Full matched emissions-driven VL/LN birth-source intervention",source_file=str(args.birth_source),
        reference="Net D_Fland-D_Eluc, all native land classes/pools excluding permafrost",time_support="2023..2099 native annual means; cumulative starts from zero before 2023",
        conditional_effect="11-10 changes future-born production response while keeping existing-forest response on; it includes all coupled state/environment responses",
        loss_remainder="NPP minus net is an aggregate algebraic remainder, not a new independently measured loss pathway or age-specific soil/product attribution")
    result.to_netcdf(args.output/"annual_contrasts_all500.nc",engine="h5netcdf",encoding={v:{"zlib":True,"complevel":4} for v in result.data_vars})
    pd.DataFrame(stats).to_csv(args.output/"annual_contrast_summary.csv",index=False)
    ends=pd.DataFrame(end_rows);ends.to_csv(args.output/"member_cumulative_net_endpoints.csv",index=False)
    stg=pd.DataFrame(stage_rows);stg.to_csv(args.output/"member_stage_net_contrasts.csv.gz",index=False,compression="gzip")
    endpoint_summary=[]
    for (bg,name),sub in ends.groupby(["scenario","contrast"],sort=False):
        for metric in [c for c in sub if c.startswith("cumulative_")]:endpoint_summary.append(dict(scenario=bg,contrast=name,metric=metric,**describe(sub[metric])))
    pd.DataFrame(endpoint_summary).to_csv(args.output/"cumulative_endpoint_summary.csv",index=False)
    stage_summary=[]
    for (bg,name,j),sub in stg.groupby(["scenario","contrast","stage_index"],sort=False):
        for metric in [c for c in sub if c.endswith(("_stage_mean","_stage_integral"))]:stage_summary.append(dict(scenario=bg,contrast=name,stage_index=int(j),metric=metric,**describe(sub[metric])))
    pd.DataFrame(stage_summary).to_csv(args.output/"stage_contrast_summary.csv",index=False)
    summary=dict(status="complete",completed_at_utc=datetime.now(timezone.utc).isoformat(),source_attributes=dict(d.attrs),
        contrasts={k:f"{a}-{b}" for k,(a,b) in CONTRASTS.items()},member_policy="All original 500, paired first within condition, no resampling or reweighting",
        stages="Age-on member-stage annual ranges from the matched-future analysis; stage definitions and populations are retained",scenarios={})
    for bg in ("VL","LN"):
        z=ends[ends.scenario==bg]
        obj={}
        for point in ("on_max","2099"):
            total=z[z.contrast=="total_on_minus_off"][f"cumulative_net_PgC_to_{point}"].mean()
            future=z[z.contrast=="future_response_existing_on"][f"cumulative_net_PgC_to_{point}"].mean()
            remaining=z[z.contrast=="remaining_existing_response"][f"cumulative_net_PgC_to_{point}"].mean()
            obj[point]=dict(total_net_mean_PgC=float(total),future_conditional_net_mean_PgC=float(future),remaining_net_mean_PgC=float(remaining),
                fraction_of_mean_total=future/total if total!=0 else None)
        summary["scenarios"][bg]=obj
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");shutil.copy2(__file__,args.output/"source_used.py")
    if args.detailed_age_source:
        with xr.open_dataset(args.detailed_age_source) as ds:
            fine=ds[["age_area_Mha","age_direct_NPP_PgC_yr"]].load()
        if fine.sizes["config"]!=500 or not np.array_equal(fine.source_config.values.astype(int),ids):
            raise ValueError("Detailed-age source IDs differ")
        if not np.array_equal(fine.year.values.astype(int),years):raise ValueError("Detailed-age annual support differs")
        fine.attrs.update(source_file=str(args.detailed_age_source),
            plotted_geometry="Widths and heights are separate ratios of ensemble/calendar means; rectangle area exactly equals mean direct NPP adjustment",
            direct_NPP="qA(g-1) at the realized age-on state/environment, not complete on-off NPP or net uptake",
            primary="Age-unclassified primary forest has zero direct adjustment by model construction, not an inferred ecological zero")
        fine.to_netcdf(args.output/"landscape_production_source_all500.nc",engine="h5netcdf",
                       encoding={v:{"zlib":True,"complevel":4} for v in fine.data_vars})
        member_rows=[];mosaic_rows=[]
        for bg in fine.scenario.values:
            f=fine.sel(scenario=bg).mean("year")
            all_forest=f.age_area_Mha.sel(detailed_age_group=sum(MOSAIC_GROUPS.values(),[])).sum("detailed_age_group")
            resolved=all_forest-f.age_area_Mha.sel(detailed_age_group="primary_unaged")
            total_direct=f.age_direct_NPP_PgC_yr.sum("detailed_age_group")
            for group,classes in MOSAIC_GROUPS.items():
                area=f.age_area_Mha.sel(detailed_age_group=classes).sum("detailed_age_group").values
                npp=f.age_direct_NPP_PgC_yr.sel(detailed_age_group=classes).sum("detailed_age_group").values
                intensity=np.divide(npp*1000,area,out=np.full_like(npp,np.nan),where=area!=0)
                for i,sid in enumerate(ids):member_rows.append(dict(scenario=str(bg),member_position=i,source_config=int(sid),
                    age_class=group,mean_area_Mha=float(area[i]),mean_direct_NPP_PgC_yr=float(npp[i]),
                    area_weighted_intensity_MgC_ha_yr=float(intensity[i]),area_defined=bool(area[i]!=0)))
                ma=float(area.mean());mn=float(npp.mean())
                mosaic_rows.append(dict(scenario=str(bg),age_class=group,mean_area_Gha=ma/1000,
                    height_MgC_ha_yr=mn*1000/ma if ma!=0 else np.nan,mean_direct_NPP_PgC_yr=mn,
                    fraction_of_mean_all_forest_area=ma/float(all_forest.mean()),
                    fraction_of_mean_age_resolved_forest_area=ma/float(resolved.mean()),
                    fraction_of_mean_direct_adjustment=mn/float(total_direct.mean()),
                    member_intensity_median=describe(intensity)["median"],
                    member_intensity_q05=describe(intensity)["q05"],member_intensity_q95=describe(intensity)["q95"]))
        pd.DataFrame(member_rows).to_csv(args.output/"member_landscape_production_geometry.csv.gz",index=False,compression="gzip")
        pd.DataFrame(mosaic_rows).to_csv(args.output/"landscape_production_geometry.csv",index=False)
    print(json.dumps(summary["scenarios"]))


if __name__=="__main__":main()

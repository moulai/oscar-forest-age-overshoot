#!/usr/bin/env python3
"""Scientific outcomes for matched age-delay temperature compensation.

The initial baseline mode identifies an already-realized window maximum that
no later intervention can undo. It reads existing completed thin temperature
sources only; no model is run and no member is selected or resampled.
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


def describe(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    if not a.size:return dict(n=0,mean=None,median=None,q05=None,q95=None)
    q=np.quantile(a,[.05,.5,.95])
    return dict(n=int(a.size),mean=float(a.mean()),median=float(q[1]),q05=float(q[0]),q95=float(q[2]),
        positive_fraction=float((a>0).mean()),negative_fraction=float((a<0).mean()),zero_fraction=float((a==0).mean()))


def last_max_year(values,years):
    return years[len(years)-1-np.argmax(values[::-1],axis=0)]


def baseline_reachability(source,first_affected,output):
    with xr.open_dataset(source) as d:
        t=d.T_1850_1900_K.sel(scenario=["VL","LN"],selector=["00","11"]).load()
        ids=d.source_config.values.copy()
        attrs=dict(d.attrs)
    if t.sizes["config"]!=500:raise ValueError("Original complete matched members are required")
    years=t.year.values.astype(int)
    if not np.array_equal(years,np.arange(2022,2100)):
        raise ValueError("The historical 2022 anchor and native futures through 2099 are required")
    members=[];summary=[]
    for scenario in ("VL","LN"):
        on=t.sel(scenario=scenario,selector="11").transpose("year","config").values
        off=t.sel(scenario=scenario,selector="00").transpose("year","config").values
        target=off.max(axis=0)
        target_year=last_max_year(off,years)
        on_max_year=last_max_year(on,years)
        off_true_peak=(off.max(axis=0)>1.5)&(target_year>years[0])&(target_year<years[-1])
        on_true_peak=(on.max(axis=0)>1.5)&(on_max_year>years[0])&(on_max_year<years[-1])
        for timing,label in first_affected.items():
            prior=years<int(label)
            if not prior.any():raise ValueError("At least the common historical anchor must precede treatment")
            fixed_max=on[prior].max(axis=0)
            fixed_year=last_max_year(on[prior],years[prior])
            bound=fixed_max-target
            excluded=bound>0
            for i,sid in enumerate(ids):
                members.append(dict(scenario=scenario,timing=timing,member_position=i,source_config=int(sid),
                    first_affected_annual_label=int(label),last_unaffected_annual_label=int(years[prior][-1]),
                    target_off_window_max_K=float(target[i]),target_off_window_max_year=int(target_year[i]),
                    off_resolved_overshoot_peak=bool(off_true_peak[i]),on_resolved_overshoot_peak=bool(on_true_peak[i]),
                    prior_on_window_max_K=float(fixed_max[i]),prior_on_window_max_year=int(fixed_year[i]),
                    irreversible_lower_bound_on_peak_residual_K=float(bound[i]),
                    peak_target_ruled_out_by_past_maximum=bool(excluded[i]),
                    not_ruled_out_is_not_proof_of_reachability=True))
            summary.append(dict(scenario=scenario,timing=timing,first_affected_annual_label=int(label),
                n_original_members=int(len(ids)),ruled_out_n=int(excluded.sum()),ruled_out_fraction=float(excluded.mean()),
                not_ruled_out_n=int((~excluded).sum()),
                lower_bound_all500_K=describe(bound),positive_lower_bound_K=describe(bound[excluded])))
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(members).to_csv(output/"member_preintervention_peak_bounds_all500.csv",index=False)
    result=dict(status="complete_existing_baseline_analysis",completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source),source_attributes=attrs,first_affected_annual_labels=first_affected,
        temporal_definition="Annual forcing for native label y is consumed in both substeps y-1 to y; labels strictly before first_affected are unchanged",
        target="Each member's all-response-off maximum across native labels 2022..2099, with common H1 historical anchor",
        bound="max(T_on, labels before treatment) - max(T_off, all labels); if strictly positive, no later intervention can undo the past annual maximum",
        caveat="The complement is only not excluded by this bound; it is not a claim that a finite dose attains the target. Late runs and all original IDs remain included.",
        member_policy="Original 500 within each path/timing, paired first, no resampling, reweighting or success filtering",
        conditions=summary)
    (output/"preintervention_peak_bounds_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    shutil.copy2(__file__,output/"analysis_source_used.py")
    print(json.dumps(summary))


def outcomes(temperature,years):
    """Use unchanged annual-window and return definitions; undefined r stays NaN."""
    temperature=np.asarray(temperature,dtype=float)
    above=temperature>1.5
    ever=above.any(axis=0)
    p=last_max_year(temperature,years)
    last_above=np.where(above,np.arange(len(years))[:,None],-1).max(axis=0)
    returned=ever&(last_above<len(years)-1)
    r=np.full(temperature.shape[1],np.nan)
    r[returned]=years[last_above[returned]+1]
    future=years>=2023
    state=np.where(~ever,"no_observed_overshoot",np.where(returned,"returned_by_2099","not_returned_by_2099"))
    return dict(window_max_K=temperature.max(axis=0),window_max_year=p,
        resolved_overshoot_peak=ever&(p>years[0])&(p<years[-1]),
        return_year=r,state=state,left_truncated_upcrossing=above[0],
        N77_years=above[future].sum(axis=0),B77_Kyr=np.maximum(temperature[future]-1.5,0).sum(axis=0),
        T2099_K=temperature[-1])


def total(data):
    dims=[d for d in data.dims if d not in ("year","config")]
    return data.sum(dims) if dims else data


def return_goal(dose,reference):
    eligible=np.isfinite(reference["return_year"])
    no_overshoot=dose["state"]=="no_observed_overshoot"
    no_later=np.isfinite(dose["return_year"])&(dose["return_year"]<=reference["return_year"])
    goal=np.where(eligible,no_overshoot|no_later,np.nan)
    labels=np.full(len(eligible),"not_applicable_reference_no_return",dtype=object)
    labels[eligible&no_overshoot]="overshoot_eliminated"
    labels[eligible&~no_overshoot&no_later]="returned_no_later"
    labels[eligible&np.isfinite(dose["return_year"])&~no_later]="returned_later"
    labels[eligible&(dose["state"]=="not_returned_by_2099")]="not_returned_by_2099"
    return eligible,goal,labels


def analyze_formal_cells(plan_path,baseline_source,output):
    plan=json.loads(plan_path.read_text())
    with xr.open_dataset(baseline_source) as src:
        base=src.sel(scenario=["VL","LN"],selector=["00","11"]).load()
    ids=base.source_config.values.astype(int)
    if not np.array_equal(ids,np.asarray(plan["source_config"],dtype=int)):
        raise ValueError("Compensation PLAN and matched baseline IDs differ")
    years=base.year.values.astype(int);future_years=base.future_year.values.astype(int)
    refs={};offsets={}
    for bg in ("VL","LN"):
        refs[bg]={}
        for name,selector in (("zero_on","11"),("target_off","00")):
            x=base.sel(scenario=bg,selector=selector)
            refs[bg][name]=outcomes(x.T_1850_1900_K.transpose("year","config").values,years)
            refs[bg][name]["cumulative_net_uptake_PgC"]=x.net_land_uptake_PgC_yr.sum("future_year").values
            refs[bg][name]["cumulative_NPP_PgC"]=x.NPP_PgC_yr.sum("future_year").values
            refs[bg][name]["CO2_2099_anomaly_ppm"]=x.CO2_anomaly_ppm.sel(year=2099).values
        on_path=next(c["baseline_paths"]["on"] for c in plan["cells"] if c["scenario"]==bg)
        with xr.open_dataset(Path(on_path)/"annual_summary.nc") as d:
            # Future output's initial 2022 label is an instantaneous state;
            # the thin 2022 anchor is the historical annual mean. Derive the
            # PI offset from matching native future annual support instead.
            first_future=int(future_years[0])
            offsets[bg]=d.D_Tg.sel(year=first_future).values-base.T_1850_1900_K.sel(scenario=bg,selector="11",year=first_future).values
    bounds=pd.read_csv(output/"member_preintervention_peak_bounds_all500.csv")
    rows=[];coverage=[];annual=[]
    for cell in plan["cells"]:
        directory=Path(cell["output"]);report_file=directory/"report.json"
        report=json.loads(report_file.read_text()) if report_file.exists() else {}
        formal=report.get("status")=="passed_temperature_compensation"
        coverage.append(dict(cell=cell["cell"],scenario=cell["scenario"],timing=cell["timing"],dose_PgC=cell["dose_PgC"],
            formally_published=formal,report_status=report.get("status","not_formally_published"),source=str(directory)))
        if not formal:continue
        bg=cell["scenario"]
        with xr.open_dataset(directory/"annual_summary.nc") as d:
            if not np.array_equal(d.source_config.values.astype(int),ids):raise ValueError("Formal cell source IDs differ")
            # Formal intervention output stores native future labels only.
            # Reuse the unchanged common H1 historical annual-2022 anchor.
            initial=base.sel(scenario=bg,selector="11",year=2022)
            T_future=d.D_Tg.sel(year=future_years).transpose("year","config").values-offsets[bg][None,:]
            co2_future=d.D_CO2.sel(year=future_years).transpose("year","config").values
            T=np.vstack((initial.T_1850_1900_K.values[None,:],T_future))
            co2=np.vstack((initial.CO2_anomaly_ppm.values[None,:],co2_future))
            net=total(d.D_Fland_net).sel(year=future_years).transpose("year","config").values
            npp=total(d.rf_NPP).sel(year=future_years).transpose("year","config").values
        dose=outcomes(T,years)
        dose["cumulative_net_uptake_PgC"]=net.sum(axis=0)
        dose["cumulative_NPP_PgC"]=npp.sum(axis=0)
        dose["CO2_2099_anomaly_ppm"]=co2[-1]
        ref=refs[bg]["target_off"];zero=refs[bg]["zero_on"]
        eligible,rgoal,rlabels=return_goal(dose,ref)
        bp=bounds[(bounds.scenario==bg)&(bounds.timing==cell["timing"])].sort_values("member_position")
        if not np.array_equal(bp.source_config.values,ids):raise ValueError("Baseline bound IDs differ")
        peak_met=dose["window_max_K"]<=ref["window_max_K"]
        n_met=dose["N77_years"]<=ref["N77_years"]
        b_met=dose["B77_Kyr"]<=ref["B77_Kyr"]
        for i,sid in enumerate(ids):
            rec=dict(cell=cell["cell"],scenario=bg,timing=cell["timing"],dose_PgC=cell["dose_PgC"],member_position=i,source_config=int(sid),
                reference_is_returner=bool(eligible[i]),reference_return_outcome=str(rlabels[i]),reference_return_goal_met=rgoal[i],
                peak_target_met=bool(peak_met[i]),N77_target_met=bool(n_met[i]),B77_target_met=bool(b_met[i]),
                joint_P_N_B_target_met=bool(peak_met[i]&n_met[i]&b_met[i]),
                joint_P_reference_return_met=float(peak_met[i] and rgoal[i]==1) if eligible[i] else np.nan,
                joint_P_N_B_reference_return_met=float(peak_met[i]&n_met[i]&b_met[i] and rgoal[i]==1) if eligible[i] else np.nan,
                peak_target_ruled_out_by_past_maximum=bool(bp.iloc[i].peak_target_ruled_out_by_past_maximum),
                irreversible_lower_bound_on_peak_residual_K=float(bp.iloc[i].irreversible_lower_bound_on_peak_residual_K))
            for label,result in (("dose_on",dose),("zero_on",zero),("target_off",ref)):
                for metric,value in result.items():
                    v=value[i]
                    rec[label+"_"+metric]=str(v) if isinstance(v,(str,np.str_)) else v.item() if isinstance(v,np.generic) else v
            for metric in ("window_max_K","N77_years","B77_Kyr","T2099_K","cumulative_net_uptake_PgC","cumulative_NPP_PgC","CO2_2099_anomaly_ppm"):
                rec["residual_"+metric+"_vs_off"]=float(dose[metric][i]-ref[metric][i])
                rec["effect_"+metric+"_vs_zero_on"]=float(dose[metric][i]-zero[metric][i])
            rec["return_year_difference_exact_vs_off"]=float(dose["return_year"][i]-ref["return_year"][i])
            rows.append(rec)
        annual.append(xr.Dataset({
            "T_1850_1900_K":(("cell","year","config"),T[None]),
            "CO2_anomaly_ppm":(("cell","year","config"),co2[None]),
            "net_land_uptake_PgC_yr":(("cell","future_year","config"),net[None]),
            "NPP_PgC_yr":(("cell","future_year","config"),npp[None])},
            coords=dict(cell=[cell["cell"]],year=years,future_year=future_years,config=base.config.values,
                source_config=("config",ids),scenario=("cell",[bg]),timing=("cell",[cell["timing"]]),dose_PgC=("cell",[cell["dose_PgC"]]))))
    pd.DataFrame(coverage).to_csv(output/"formal_coverage.csv",index=False)
    effects=pd.DataFrame(rows)
    summaries=[];transitions=[];timing_rows=[]
    if not effects.empty:
        effects.to_csv(output/"member_compensation_outcomes_all500.csv",index=False)
        for cell,s in effects.groupby("cell",sort=False):
            key={k:s.iloc[0][k] for k in ("cell","scenario","timing","dose_PgC")}
            for metric in [c for c in s if c.startswith(("residual_","effect_"))]+["return_year_difference_exact_vs_off"]:
                population="both_dose_and_reference_returned" if metric=="return_year_difference_exact_vs_off" else "all_original_members"
                summaries.append(dict(**key,population=population,metric=metric,**describe(s[metric])))
            for metric in ("peak_target_met","N77_target_met","B77_target_met","joint_P_N_B_target_met"):
                summaries.append(dict(**key,population="all_original_members",metric=metric,n=len(s),mean=float(s[metric].mean()),
                    median=None,q05=None,q95=None))
            rr=s[s.reference_is_returner]
            for metric in ("reference_return_goal_met","joint_P_reference_return_met","joint_P_N_B_reference_return_met"):
                summaries.append(dict(**key,population="fixed_reference_returners",metric=metric,n=len(rr),mean=float(rr[metric].mean()),
                    median=None,q05=None,q95=None))
            for names,ss in s.groupby(["target_off_state","zero_on_state","dose_on_state"],sort=False):
                transitions.append(dict(**key,target_off_state=names[0],zero_on_state=names[1],dose_on_state=names[2],n=len(ss),fraction_original_members=len(ss)/500))
        pd.DataFrame(summaries).to_csv(output/"condition_compensation_summary.csv",index=False)
        pd.DataFrame(transitions).to_csv(output/"return_state_transitions.csv",index=False)
        for (bg,q),s in effects.groupby(["scenario","dose_PgC"],sort=False):
            if set(s.timing)!={"early","late"}:continue
            early=s[s.timing=="early"].sort_values("member_position");late=s[s.timing=="late"].sort_values("member_position")
            if not np.array_equal(early.source_config,late.source_config):raise ValueError("Timing-pair IDs differ")
            for i,sid in enumerate(ids):
                row=dict(scenario=bg,dose_PgC=q,member_position=i,source_config=int(sid))
                for metric in ("window_max_K","N77_years","B77_Kyr","T2099_K","cumulative_net_uptake_PgC","CO2_2099_anomaly_ppm"):
                    row["early_minus_late_"+metric]=float(early.iloc[i]["dose_on_"+metric]-late.iloc[i]["dose_on_"+metric])
                row["early_state"]=early.iloc[i].dose_on_state;row["late_state"]=late.iloc[i].dose_on_state
                timing_rows.append(row)
        if timing_rows:
            timing_table=pd.DataFrame(timing_rows);timing_table.to_csv(output/"member_equal_dose_timing_effects_all500.csv",index=False)
            ts=[]
            for (bg,q),s in timing_table.groupby(["scenario","dose_PgC"],sort=False):
                for metric in [c for c in s if c.startswith("early_minus_late_")]:
                    ts.append(dict(scenario=bg,dose_PgC=q,metric=metric,**describe(s[metric])))
            pd.DataFrame(ts).to_csv(output/"equal_dose_timing_summary.csv",index=False)
        cube=xr.concat(annual,dim="cell");cube.attrs.update(scientific_family="Full matched age-on non-land CO2 compensation",plan_identity=plan["plan_identity"],
            time_support="Temperature includes common 2022 anchor; NPP/net future_year2023..2099 exclude anchor",
            temperature_reference="Common H1 member-wise1850-1900 baseline; PI offset obtained from identical first-future annual support, not future initial-state2022 versus historical annual2022",
            net_definition="D_Fland_net=D_Fland-D_Eluc, all native land pools/classes excluding permafrost")
        cube.to_netcdf(output/"source_annual_compensation_all500.nc",engine="h5netcdf",encoding={v:{"zlib":True,"complevel":4} for v in cube.data_vars})
    bracket_rows=[]
    for bg in ("VL","LN"):
        z=refs[bg]["zero_on"];r=refs[bg]["target_off"]
        ref_eligible,zero_return_goal,_=return_goal(z,r)
        for timing in ("early","late"):
            b=bounds[(bounds.scenario==bg)&(bounds.timing==timing)].sort_values("member_position")
            available={}
            for q in (10,30):
                if not effects.empty:
                    ss=effects[(effects.scenario==bg)&(effects.timing==timing)&(effects.dose_PgC==q)].sort_values("member_position")
                    if len(ss)==500:available[q]=ss
            for i,sid in enumerate(ids):
                residuals={0:float(z["window_max_K"][i]-r["window_max_K"][i])}
                for q,ss in available.items():residuals[q]=float(ss.iloc[i].residual_window_max_K_vs_off)
                met=[q for q,v in residuals.items() if v<=0]
                smallest=min(met) if met else np.nan
                excluded=bool(b.iloc[i].peak_target_ruled_out_by_past_maximum)
                monotone=(residuals[0]>=residuals[10]>=residuals[30]) if len(residuals)==3 else None
                status=("past_maximum_excludes_peak_target" if excluded else "zero_dose_already_meets" if smallest==0 else
                    "sampled_positive_dose_meets" if met else "no_tested_dose_meets" if len(residuals)==3 else "awaiting_formal_dose_results")
                lower=max([q for q,v in residuals.items() if q<smallest and v>0],default=np.nan) if met and smallest>0 else np.nan
                rec=dict(scenario=bg,timing=timing,member_position=i,source_config=int(sid),status=status,
                    peak_target_ruled_out_by_past_maximum=excluded,minimum_tested_dose_meeting_peak_PgC=smallest,
                    sampled_lower_dose_not_meeting_peak_PgC=lower,sampled_upper_dose_meeting_peak_PgC=smallest if met and smallest>0 else np.nan,
                    monotone_at_all_three_tested_doses=monotone,continuous_required_dose_not_solved=True,
                    reference_is_returner=bool(ref_eligible[i]),zero_reference_return_goal_met=zero_return_goal[i])
                for q in (0,10,30):rec[f"peak_residual_at_{q}PgC_K"]=residuals.get(q,np.nan)
                bracket_rows.append(rec)
    brackets=pd.DataFrame(bracket_rows);brackets.to_csv(output/"member_target_brackets_all500.csv",index=False)
    bracket_counts=brackets.groupby(["scenario","timing","status"],sort=False).size().reset_index(name="n")
    bracket_counts["fraction_original_members"]=bracket_counts.n/500
    bracket_counts.to_csv(output/"target_bracket_counts.csv",index=False)
    accounting=json.loads((plan_path.parent/"input_accounting.json").read_text())
    pd.DataFrame(accounting["cells"]).to_csv(output/"source_input_accounting.csv",index=False)
    shutil.copy2(plan_path.parent/"annual_CO2_intervention.csv",output/"source_annual_CO2_intervention.csv")
    base.to_netcdf(output/"source_baseline_on_off_all500.nc",engine="h5netcdf",encoding={v:{"zlib":True,"complevel":4} for v in base.data_vars})
    complete=sum(c["formally_published"] for c in coverage)
    result=dict(status="complete" if complete==len(coverage) else "partial_formal_coverage",completed_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_identity=plan["plan_identity"],n_expected_cells=len(coverage),n_formally_published_cells=complete,
        original_members_per_condition=500,coverage=coverage,primary_outcomes="Paired window-maximum residual and fixed-reference-returner retention/no delay; N/B and full joint outcomes are supporting",
        return_goal="Only reference off-returners have a return-year target. Dose eliminating overshoot is separately labelled, never assigned r=0. Others retain window states.",
        dose_inference="Actual common-dose outcomes are separate from memberwise sampled brackets. No continuous inverse solution, extrapolation, new dose or model run is performed.",
        uncertainty="Original joint-parameter empirical distributions separately within each path/timing/dose, not real probabilities or independent condition-member replicates")
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    shutil.copy2(plan_path,output/"source_plan.json");shutil.copy2(__file__,output/"analysis_source_used.py")
    print(json.dumps(dict(status=result["status"],formal_cells=complete,output=str(output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source",type=Path,required=True)
    p.add_argument("--first-affected-early",type=int,required=True)
    p.add_argument("--first-affected-late",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plan",type=Path,help="Analyze only formally published compensation cells from this locked PLAN")
    a=p.parse_args()
    baseline_reachability(a.baseline_source,{"early":a.first_affected_early,"late":a.first_affected_late},a.output)
    if a.plan:analyze_formal_cells(a.plan,a.baseline_source,a.output)


if __name__=="__main__":main()

"""Factorial source responses in matched VL/LN, preserving all original IDs.

Primary comparisons are new-cohort response with existing response absent
(01-00) or present (11-10), at the same member-specific 00 reference year.
Annual data identify coupled temperature/net-carbon contrasts, not exact
four-arm radiative components or cohort-labelled soil/product net uptake.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows.analyze_future_age_response import read_arm,describe,plain,YEARS
from workflows.analyze_restoration_emissions import trajectory_endpoints

SCHEMA="matched-cmip7-birth-source-age-analysis-v1"
MODES=("00","10","01","11")


def contrasts(values):
    """Return conditional effects and an unnormalized source interaction."""
    n0=values["01"]-values["00"]
    n1=values["11"]-values["10"]
    e0=values["10"]-values["00"]
    e1=values["11"]-values["01"]
    return dict(N0=n0,N1=n1,E0=e0,E1=e1,I=n1-n0,
                total=values["11"]-values["00"],N0_minus_E0=n0-e0)


def interval_columns(row,name,interval):
    for k in ("lower","upper","lower_strict","upper_strict"):
        value=None if interval is None else getattr(interval,k)
        if isinstance(value,(float,np.floating)) and np.isinf(value):
            value="+Infinity" if value>0 else "-Infinity"
        row[name+"_"+k]=value


def analyze(args):
    plan=json.loads(args.plan.read_text())
    if plan.get("schema")!="matched-cmip7-birth-source-age-response-v1":
        raise RuntimeError("A matched-source intervention plan is required")
    ids=plan["source_config"]
    with xr.open_dataset(ROOT/plan["history_directory"]/"annual_summary.nc") as h:
        if h.source_config.values.tolist()!=ids:raise RuntimeError("Common history member order changed")
        offset=h.D_Tg.sel(year=slice(1850,1900)).mean("year").values
        anchor_t=h.D_Tg.sel(year=2022).values
        anchor_co2=h.D_CO2.sel(year=2022).values
    contrast_rows=[];arm_rows=[];backgrounds={};source_records=[];all_arrays=[]
    for bg in plan["scenarios"]:
        base=next(b for b in plan["baselines"] if b["scenario"]==bg)
        paths=dict(base["paths"])
        expected=dict(base["run_identities"])
        for mode in ("10","01"):
            cell=next(c for c in plan["cells"] if c["scenario"]==bg and c["selector"]==mode)
            path=ROOT/cell["output"]
            r=json.loads((path/"report.json").read_text())
            if r.get("status")!="passed_birth_source_response" or r.get("members")!=500 or r.get("plan_identity")!=plan["plan_identity"]:
                raise RuntimeError("A new source-selective arm is incomplete or mismatched")
            paths[mode]=cell["output"];expected[mode]=r["run_identity"]
        arrays={m:read_arm(ROOT/paths[m],ids,anchor_t,anchor_co2,offset) for m in MODES}
        if any(arrays[m]["run_identity"]!=expected[m] for m in MODES):
            raise RuntimeError("A reused or new arm changed scientific identity")
        endpoints={m:[trajectory_endpoints(YEARS,arrays[m]["T"][:,i]) for i in range(500)] for m in MODES}
        for i,sid in enumerate(ids):
            ep={m:endpoints[m][i] for m in MODES}
            p00=int(ep["00"].peak_year);index=p00-2022
            interior=2022<p00<2099
            followup=interior and p00+20<=2099
            row=dict(scenario=bg,member_position=i,source_config=sid,p00_year=p00,
                reference00_interior_maximum=interior,reference00_resolved_overshoot=ep["00"].duration_defined,
                reference00_plus20_defined=followup,
                baseline00_11_resolved_overshoot=ep["00"].duration_defined and ep["11"].duration_defined,
                all_four_resolved_overshoot=all(e.duration_defined for e in ep.values()))
            metrics={k:{} for k in ("T_at_p00_K","window_maximum_K","selection_remainder_K","gap20_K",
                                     "net_to_p00_PgC","net_2023_2099_PgC","CO2_at_p00_ppm","T2099_K")}
            for mode in MODES:
                a,e=arrays[mode],ep[mode]
                t=a["T"][:,i]
                arm=dict(scenario=bg,selector=mode,member_position=i,source_config=sid,status=e.status,
                    window_maximum_K=e.peak_K,window_maximum_year=e.peak_year,
                    resolved_overshoot_peak=e.duration_defined,p00_year=p00,T_at_p00_K=t[index],
                    T2099_K=t[-1],CO2_at_p00_ppm=a["CO2"][index,i],
                    net_to_p00_PgC=a["net"][YEARS[1:]<=p00,i].sum(),
                    net_2023_2099_PgC=a["net"][:,i].sum(),
                    reference00_plus20_defined=followup,
                    T_change_reference20_K=t[index+20]-t[index] if followup else np.nan,
                    peak_before_reference20=(int(e.peak_year)<=p00+20) if followup else None)
                for name in ("return_year","duration_years"):interval_columns(arm,name,getattr(e,name))
                arm_rows.append(arm)
                metrics["T_at_p00_K"][mode]=arm["T_at_p00_K"]
                metrics["window_maximum_K"][mode]=e.peak_K
                metrics["selection_remainder_K"][mode]=(e.peak_K-ep["00"].peak_K)-(t[index]-arrays["00"]["T"][index,i])
                metrics["gap20_K"][mode]=arm["T_change_reference20_K"]
                for name in ("net_to_p00_PgC","net_2023_2099_PgC","CO2_at_p00_ppm","T2099_K"):
                    metrics[name][mode]=arm[name]
                for name in ("status","window_maximum_year","resolved_overshoot_peak"):
                    row[mode+"_"+name]=arm[name]
            for metric,values in metrics.items():
                for label,value in contrasts(values).items():row[label+"_"+metric]=value
            contrast_rows.append(row)
        frame=pd.DataFrame([r for r in contrast_rows if r["scenario"]==bg])
        metric_cols=[k for k in frame if k.startswith(("N0_","N1_","E0_","E1_","I_","total_"))]
        population_masks={
            "all_original_500":np.ones(500,dtype=bool),
            "reference00_resolved_overshoot":frame.reference00_resolved_overshoot.to_numpy(),
            "reference00_plus20_defined":frame.reference00_plus20_defined.to_numpy(),
            "reference00_resolved_plus20":(frame.reference00_resolved_overshoot&frame.reference00_plus20_defined).to_numpy(),
            "baseline00_11_resolved_overshoot":frame.baseline00_11_resolved_overshoot.to_numpy(),
            "all_four_resolved_overshoot":frame.all_four_resolved_overshoot.to_numpy()}
        summaries={}
        for name,use in population_masks.items():
            chosen=frame.loc[use]
            summaries[name]=dict(n=int(use.sum()),metrics={k:describe(chosen[k]) for k in metric_cols})
            for metric in ("T_at_p00_K","gap20_K","net_to_p00_PgC"):
                v=chosen[["N0_"+metric,"N1_"+metric,"E0_"+metric,"E1_"+metric]]
                available=v.notna().all(axis=1)
                v=v.loc[available]
                summaries[name].setdefault("source_direction_counts",{})[metric]=dict(n=len(v),
                    both_newborn_effects_positive=int(((v["N0_"+metric]>0)&(v["N1_"+metric]>0)).sum()),
                    future_only_larger_than_existing_only=int((v["N0_"+metric]>v["E0_"+metric]).sum()),
                    existing_only_larger_than_future_only=int((v["N0_"+metric]<v["E0_"+metric]).sum()))
        backgrounds[bg]=dict(original_members=500,populations=summaries,
            status_counts={m:dict(Counter(e.status for e in endpoints[m])) for m in MODES})
        source_records.append(dict(scenario=bg,paths=paths,run_identities=expected))
        all_arrays.append(arrays)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(contrast_rows).to_csv(args.output/"member_source_contrasts_all500.csv",index=False)
    pd.DataFrame(arm_rows).to_csv(args.output/"arm_members_all500.csv",index=False)
    ds=xr.Dataset(coords=dict(scenario=plan["scenarios"],selector=list(MODES),year=YEARS,
        future_year=YEARS[1:],config=np.arange(500),source_config=("config",ids)))
    for target,name,dimension in (("T_1850_1900_K","T","year"),("CO2_anomaly_ppm","CO2","year"),
                                  ("net_land_uptake_PgC_yr","net","future_year"),("NPP_PgC_yr","NPP","future_year")):
        ds[target]=(("scenario","selector",dimension,"config"),
                    np.stack([np.stack([a[m][name] for m in MODES]) for a in all_arrays]))
    ds.attrs.update(schema=SCHEMA,plan_identity=plan["plan_identity"],history_run_identity=plan["history_run_identity"],
        time_support="Common historical annual2022 anchor; native annual means2023-2099; integrals exclude2022",
        temperature_reference="Common H1 member-wise1850-1900 mean",
        net_land_definition="D_Fland-D_Eluc; all native land classes/pools excluding permafrost")
    ds.to_netcdf(args.output/"source_annual_four_arms_all500.nc",engine="netcdf4",
                 encoding={k:dict(zlib=True,complevel=2) for k in ds.data_vars})
    result=dict(schema=SCHEMA,status="complete",generated_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_identity=plan["plan_identity"],history_run_identity=plan["history_run_identity"],
        comparison_family=plan["comparison_family"],scenarios=plan["scenarios"],members_per_condition=500,
        new_scientific_arms=4,new_member_trajectories=2000,contrast_rows=len(contrast_rows),arm_rows=len(arm_rows),
        definitions=dict(N0="01-00: future-born production response with existing response off",
            N1="11-10: future-born production response with existing response on",
            E0="10-00: existing response with future-born response off",
            E1="11-01: existing response with future-born response on",
            I="N1-N0 = 11-10-01+00, computed per ID",
            total="11-00",N0_minus_E0="01-10: future-only minus existing-only; N1-E1 is algebraically identical, not independent corroboration",
            common_year="Each original member's00 window-maximum year p00",
            selection="Window-maximum contrast minus same-p00 temperature contrast; not an independent peak-time cause",
            gap20="Difference in signed temperature change from p00 to p00+20; original00 eligibility",
            net="Native2023 prefix to p00 or2099; PgC uptake; no2022 integration",
            reference="Prescribed host production without age modulation; not all age-blind models or a mature-forest truth"),
        sources=source_records,populations=backgrounds,
        limits=dict(source_interaction="Coupled model interventions with a conditional reference; no enforced100percent shares",
            origin="Actual birth_time; not carbon-atom ancestry; primary unaged",
            diagnostics="New direct outputs use consumed masked g; old11 lacks native diagnostics",
            radiative="No exact four-arm ERF-temperature attribution from annual means",
            support="All original500 retained; actual peaks, follow-up support and censoring reported separately",
            uncertainty="Within-condition empirical joint-parameter distributions, not pooled pathway confidence intervals"))
    (args.output/"summary.json").write_text(json.dumps(plain(result),indent=2,allow_nan=False)+"\n")
    shutil.copyfile(Path(__file__),args.output/"analysis_source_used.py")
    (args.output/"README.md").write_text("""# Birth-source age responses in matched VL and LN

00 and11 are reused full, unretimed futures at the same H1 emissions2022 state. Only10(existing cohorts) and01(post2022 births) are new scientific arms. Original500 IDs and joint parameters remain ordered within every condition.

Primary contrasts N0=01-00 and N1=11-10 hold existing-cohort treatment fixed within each comparison. Their future-born cohorts have no area at the2022 anchor; divergence follows actual subsequent establishment. E0/E1 and the factorial interaction are retained. Results are conditional on the prescribed unmodulated host-production reference, not a universal warming bias of all age-blind models.

member_source_contrasts_all500.csv has1000 rows (two pathways times500). arm_members_all500.csv has4000 rows including reused00/11. Every status, interval and support flag is retained. Source annual NC preserves all four trajectories. The p00 contrast precedes each-arm maximum comparisons; selection remainders are descriptive, not peak-time causal fractions. No100percent temperature allocation is imposed.

New native diagnostics record actual masked production and forcing. Existing11 has no matching native output, so annual T/net contrasts are valid but exact four-arm radiative attribution is not reported.
""")
    print(json.dumps(dict(status="complete",contrast_rows=len(contrast_rows),arm_rows=len(arm_rows))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    analyze(p.parse_args())


if __name__=="__main__":main()

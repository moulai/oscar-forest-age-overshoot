"""Prepare/run the two source-selective age responses in matched VL and LN.

Only 10 and 01 are new scientific arms. Existing full-pathway 00/11 are reused.
The production selector wraps the normal labeled age evaluator; no core model,
carbon routing, environment, age clock or calibration equation is changed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS"):
    os.environ[key]="1"
os.environ["PYTHONDONTWRITEBYTECODE"]="1"

import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from workflows.run_future_age_response import (
    now,digest,identity,write_json,local,merge_stream,add_production,validate_sampling)

SCHEMA="matched-cmip7-birth-source-age-response-v1"
SCENARIOS=("VL","LN")
SELECTORS=("10","01")
ARM_NAMES={"10":"existing_only","01":"post2022_only"}
BASE_PLAN=ROOT/"outputs/age_module/future_age_response/PLAN.json"
DEFAULT_OUTPUT=ROOT/"outputs/age_module/birth_source_age_response"
CUTOFF=2022.


class SourceSelectedProduction:
    """Use the original W/F evaluator only for a fixed actual birth source."""
    def __init__(self,kernel,births,selector):
        if selector not in SELECTORS:
            raise ValueError("A partial selector must be 10 or 01")
        self.kernel=kernel
        self.region_id=kernel.region_id
        self.mask=(np.asarray(births)<=CUTOFF if selector=="10"
                   else np.asarray(births)>CUTOFF)

    def factors(self,age,*,member_ids):
        values=self.kernel.factors(age,member_ids=member_ids)
        if values.shape[-1]!=len(self.mask):
            raise ValueError("Birth slots differ from the fixed source selector")
        result=values.copy()
        result[...,~self.mask]=1.
        return result


def selected_production(kernels,births,selector):
    """Preserve exact original full-on/off branches; wrap only the partials."""
    if selector=="00":
        return None
    if kernels is None:
        raise ValueError("An enabled source requires the original age evaluators")
    if selector=="11":
        return kernels
    if selector not in SELECTORS:
        raise ValueError("Unknown birth-source selector")
    return tuple(SourceSelectedProduction(k,births,selector) for k in kernels)


def source_versions():
    names=("run_birth_source_age_response.py","analyze_birth_source_age_response.py",
           "run_future_age_response.py","run_age_peak_mechanism_diagnostics.py",
           "prepare_regional_forest_age_inputs.py","run_regional_forest_age_future.py",
           "analyze_future_age_response.py")
    return {name:digest(ROOT/"workflows"/name) for name in names}


def prepare(args):
    base=json.loads(BASE_PLAN.read_text())
    execution=json.loads((ROOT/base["output_root"]/"execution_manifest.json").read_text())
    if execution.get("status")!="complete" or execution.get("completed_member_trajectories")!=2500:
        raise RuntimeError("The five matched-future off arms must already be complete")
    ids=base["source_config"]
    if len(ids)!=500 or len(set(ids))!=500:
        raise RuntimeError("The original ordered 500 are required")
    baselines=[];cells=[]
    for bg in SCENARIOS:
        original=next(c for c in base["cells"] if c["scenario"]==bg)
        paths={"00":original["output"],"11":original["comparison_on"]}
        reports={}
        for mode,path in paths.items():
            report=json.loads((ROOT/path/"report.json").read_text())
            status="passed_future" if mode=="11" else "passed_future_age_response"
            if report.get("status")!=status or report.get("members")!=500:
                raise RuntimeError("A reusable baseline is incomplete: "+bg+"/"+mode)
            if report.get("history_run_identity",report.get("configuration",{}).get("history_run_identity"))!=base["history_run_identity"]:
                raise RuntimeError("A reusable baseline has a different history")
            with xr.open_dataset(ROOT/path/"annual_summary.nc") as annual:
                if annual.source_config.values.tolist()!=ids:
                    raise RuntimeError("A reusable baseline changed member order")
            reports[mode]=report
        if reports["11"]["run_identity"]!=original["comparison_on_identity"]:
            raise RuntimeError("The completed H1-on identity changed")
        baselines.append(dict(scenario=bg,paths=paths,
            run_identities={m:r["run_identity"] for m,r in reports.items()}))
        for selector in SELECTORS:
            cells.append(dict(scenario=bg,selector=selector,arm=ARM_NAMES[selector],
                output=local(args.output_root/bg/ARM_NAMES[selector]),
                forcing=original["forcing"],forcing_sha256=original["forcing_sha256"],
                requests=original["requests"],requests_sha256=original["requests_sha256"],
                inherited_land_driver_treatment=original["inherited_land_driver_treatment"]))
    plan=dict(schema=SCHEMA,comparison_family="Matched VL/LN CMIP7, unretimed activity, birth-source production intervention",
        scenarios=list(SCENARIOS),selectors=list(SELECTORS),members_per_arm=500,source_config=ids,
        ordered_source_ids_sha256=identity(ids),anchor_year=2022,end_year=2099,
        history_directory=base["history_directory"],history_run_identity=base["history_run_identity"],
        parent_plan_identity=base["plan_identity"],output_root=local(args.output_root),
        solver=base["solver"],birth_source_cutoff=CUTOFF,
        selector_definition={"10":"Original gW/gF only for birth_time<=2022; other birth slots use one",
                             "01":"Original gW/gF only for birth_time>2022; other birth slots use one",
                             "00":"Reuse original production-off; native age_production=None",
                             "11":"Reuse original H1-on; unchanged age evaluators"},
        primary_rule="Unaged primary remains unmodulated in every arm; no age or carbon reassignment",
        baselines=baselines,cells=cells,source_versions=source_versions())
    plan["plan_identity"]=identity(plan)
    plan["prepared_at_utc"]=now()
    path=args.output_root/"PLAN.json"
    if path.exists():
        if json.loads(path.read_text()).get("plan_identity")!=plan["plan_identity"]:
            raise FileExistsError("Refusing to replace a different source-intervention plan")
        print(json.dumps(dict(status="already_prepared_not_run",plan=local(path),plan_identity=plan["plan_identity"])))
        return
    write_json(path,plan)
    print(json.dumps(dict(status="prepared_not_run",plan=local(path),plan_identity=plan["plan_identity"],
                          new_scientific_arms=4,member_trajectories=2000)))


def load_plan(path):
    p=json.loads(path.read_text())
    if p.get("schema")!=SCHEMA or p.get("scenarios")!=list(SCENARIOS) or p.get("selectors")!=list(SELECTORS):
        raise RuntimeError("Not the matched VL/LN four-arm source intervention")
    if p.get("source_versions")!=source_versions():
        raise RuntimeError("Prepared implementation changed; do not silently run a different source")
    if p.get("members_per_arm")!=500 or len(p["source_config"])!=500:
        raise RuntimeError("Original 500 members are required")
    return p


def interface_check(args):
    """Use actual first-ten curves; no model interval or scientific run."""
    from workflows import run_regional_forest_age_future as future
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    parameters,layout,kernels,_=load_runtime_inputs(future.INPUTS,births=future.BIRTHS,
        tau_years=future.TAU_YEARS,age_enabled=True,member_slice=slice(0,10))
    ages=np.maximum(2022.25-np.asarray(layout.births),0.)
    if selected_production(kernels,layout.births,"11") is not kernels or selected_production(kernels,layout.births,"00") is not None:
        raise RuntimeError("Full on/off branches changed")
    checked=[]
    for mode in SELECTORS:
        selected=selected_production(kernels,layout.births,mode)
        for original,wrapped in zip(kernels,selected):
            raw=original.factors(ages,member_ids=layout.parameter_ids)
            actual=wrapped.factors(ages,member_ids=layout.parameter_ids)
            if (actual.shape!=raw.shape or wrapped.region_id!=original.region_id
                    or not np.array_equal(actual[...,wrapped.mask],raw[...,wrapped.mask])
                    or not np.all(actual[...,~wrapped.mask]==1.)):
                raise RuntimeError("Source selection changed an enabled value or failed to mask")
        checked.append(mode)
    record=dict(schema=SCHEMA,status="interface_passed_no_model_integration",checked_selectors=checked,
        regions=len(layout.regions),members=np.asarray(layout.parameter_ids).tolist(),birth_slots=len(layout.births),
        source_cutoff=CUTOFF,first_future_birth=float(np.asarray(layout.births)[np.asarray(layout.births)>CUTOFF].min()),
        solver_intervals_executed=0,original_on_off_branches_preserved=True,source_versions=source_versions())
    write_json(args.output_root/"selector_interface_check.json",record)
    print(json.dumps(record))


def worker(args):
    from workflows import run_regional_forest_age_future as future
    from workflows.run_age_peak_mechanism_diagnostics import Observer,netcdf
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    from workflows.run_regional_forest_age import SUMMARIES,diagnostics,drop_prescribed_co2
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES,HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_secondary_carbon import SEC_EXCHANGES,PROJECTION
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    plan=load_plan(args.plan)
    cell=next(c for c in plan["cells"] if c["scenario"]==args.scenario and c["selector"]==args.selector)
    selection=slice(args.member_start,args.member_stop);ids=plan["source_config"][selection]
    history_dir=ROOT/plan["history_directory"]
    if future.history_report("H1","emissions")["run_identity"]!=plan["history_run_identity"]:
        raise RuntimeError("The common historical state changed")
    with xr.open_dataset(history_dir/"final_state.nc") as s:state=s.isel(config=selection).load()
    with xr.open_dataset(history_dir/"parameters.nc") as s:parameters=s.isel(config=selection).load()
    with xr.open_dataset(ROOT/cell["forcing"]) as s:forcing=s.isel(config=selection,missing_dims="ignore").load()
    with xr.open_dataset(ROOT/cell["requests"]) as s:
        requests=s[["Q_sec_harvest","Q_sec_harvest_by_label"]].isel(config=selection,missing_dims="ignore").load()
    if "Q_sec_harvest" not in forcing:forcing=xr.merge([forcing,requests],join="exact",compat="no_conflicts")
    for value in (state,parameters,forcing):
        if value.source_config.values.tolist()!=ids:raise RuntimeError("An original ID/order changed")
    forcing=future.slice_forcing(forcing,slice(None),2099)
    future.check_restart_anchor(state,forcing,"emissions")
    forcing=drop_prescribed_co2(forcing)
    _,layout,kernels,_=load_runtime_inputs(future.INPUTS,births=future.BIRTHS,
        tau_years=future.TAU_YEARS,age_enabled=True,member_slice=selection)
    if not np.array_equal(state.birth_time.values,np.asarray(layout.births)):
        raise RuntimeError("The common-state birth slots differ")
    layout.validate(state,2022.)
    selected=selected_production(kernels,layout.births,args.selector)
    model=assemble_native_forest_age_model(layout,age_production=selected,age_clock=None,
        secondary_agb_events=future.SECONDARY_HARVEST,harvest_projection=PROJECTION)
    out=args.batch_output;out.mkdir(parents=True,exist_ok=False)
    run_id=identity(dict(plan_identity=plan["plan_identity"],scenario=args.scenario,selector=args.selector))
    report=dict(schema=SCHEMA,status="running",run_identity=run_id,plan_identity=plan["plan_identity"],
        scenario=args.scenario,selector=args.selector,member_start=args.member_start,member_stop=args.member_stop,
        source_config=ids,history_run_identity=plan["history_run_identity"],source_versions=source_versions())
    write_json(out/"status.json",report)
    observer=Observer(model,layout,parameters,2099);started=time.perf_counter()
    try:
        observer.install()
        retained=list(dict.fromkeys(list(SUMMARIES)+list(AGE_EXCHANGES)+list(HARVEST_EXCHANGES)+list(SEC_EXCHANGES)
            +["D_CO2","D_CH4","D_Tg","D_Td","D_Pg","D_Tl","r_Pl","D_Ebb_CH4"]))
        annual_all,final=model(state,parameters,forcing,dtype=np.float64,var_keep=retained,
            keep_prog=False,get_final=True,nt=2,adapt_nt=False,scheme="imex")
        native,direct=observer.datasets(final)
    finally:observer.restore()
    if int(final.year)!=2099:raise RuntimeError("Future did not reach 2099")
    budget=diagnostics(state,final,annual_all)
    solved=future.assert_solved_atmosphere(annual_all,forcing)
    annual=annual_all.sel(year=slice(2023,2099))
    annual["D_Eff"]=forcing.D_Eff.sel(year=annual.year)
    annual["D_Fland_net"]=annual.D_Fland-annual.D_Eluc
    annual.D_Fland_net.attrs.update(units="PgC yr-1",definition="All native land classes/pools, D_Fland-D_Eluc, excluding permafrost")
    disabled=([g for g in direct.forest_group.values if str(g).startswith("post2022")]
              if args.selector=="10" else [g for g in direct.forest_group.values if str(g).startswith("existing")])
    disabled=list(disabled)+["primary_unaged"]
    if np.any(direct.direct_NPP_PgC_yr.sel(forest_group=disabled).values!=0.):
        raise RuntimeError("A disabled source recorded a nonzero consumed multiplier effect")
    sampling=validate_sampling(native,annual)
    add_production(annual,direct)
    for variable in ("mechanism_direct_NPP","mechanism_actual_NPP_by_origin"):
        annual[variable].attrs.update(diagnostic_schema=SCHEMA,consumed_production_selector=args.selector)
    attrs=dict(run_identity=run_id,family=SCHEMA,scenario=args.scenario,selector=args.selector,
        arm=ARM_NAMES[args.selector],history_run_identity=plan["history_run_identity"],plan_identity=plan["plan_identity"],
        birth_source_cutoff=CUTOFF,production_rule=plan["selector_definition"][args.selector],
        history_policy="area_proportional",harvest_contract="secondary_gross_agb_request_proportional",
        harvest_projection=PROJECTION,time_convention="Native annual means2023-2099; shared historical annual2022 analysis anchor")
    for dataset in (annual,native,final,parameters):dataset.attrs.update(attrs)
    for dataset,name in ((annual,"annual_summary.nc"),(native,"mechanism_native_steps.nc"),
                         (final,"final_state.nc"),(parameters,"parameters.nc")):netcdf(dataset,out/name)
    report.update(status="complete",members=len(ids),native_step_count=len(observer.records),
        masked_direct_disabled_sources=list(map(str,disabled)),sampling_reconstruction=sampling,
        diagnostics=budget,solved_atmosphere=solved,elapsed_seconds=time.perf_counter()-started)
    write_json(out/"report.json",future.plain(report));write_json(out/"status.json",future.plain(report))


def director(args):
    plan=load_plan(args.plan)
    output=ROOT/plan["output_root"]
    staging=ROOT/".runtime/birth_source_age_response"/plan["plan_identity"][:16]
    staging.mkdir(parents=True,exist_ok=args.resume)
    lock=output/".run.lock";fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=output/"execution_manifest.json"
    execution=dict(schema=SCHEMA,status="running",started_at_utc=now(),plan_identity=plan["plan_identity"],
        jobs=args.jobs,batch_size=10,new_scientific_arms=4,completed_batches=[],completed_member_trajectories=0)
    try:
        active=[]
        for cell in plan["cells"]:
            target=ROOT/cell["output"]
            if target.exists():
                r=json.loads((target/"report.json").read_text())
                if not args.resume or r.get("status")!="passed_birth_source_response" or r.get("plan_identity")!=plan["plan_identity"]:
                    raise FileExistsError("A scientific arm already exists: "+str(target))
            else:active.append((cell["scenario"],cell["selector"]))
        write_json(receipt,execution)
        jobs=[(bg,mode,i,i+10) for i in range(0,500,10) for bg,mode in active]

        def launch(job):
            bg,mode,first,last=job;key=bg+"_"+mode
            out=staging/key/"batches"/f"{first:03d}_{last:03d}"
            if out.exists():
                previous=json.loads((out/"status.json").read_text())
                if not args.resume or previous.get("plan_identity")!=plan["plan_identity"]:
                    raise RuntimeError("An unknown staging batch already exists")
                if previous.get("status")=="complete":return job
                shutil.rmtree(out)
            log=staging/"launch"/f"{key}_{first:03d}_{last:03d}.log";log.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,"-u","-B",str(Path(__file__).resolve()),"--worker","--plan",str(args.plan),
                "--scenario",bg,"--selector",mode,"--member-start",str(first),"--member-stop",str(last),"--batch-output",str(out)]
            with log.open("w") as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONHASHSEED="0",PYTHONPATH=str(ROOT)))
            if result.returncode:raise RuntimeError("Failed source-selective batch; see "+str(log))
            return job

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for result in as_completed([pool.submit(launch,j) for j in jobs]):
                bg,mode,first,last=result.result()
                execution["completed_batches"].append(dict(scenario=bg,selector=mode,first=first,last=last))
                write_json(receipt,execution)
        for cell in plan["cells"]:
            bg,mode=cell["scenario"],cell["selector"]
            if (bg,mode) not in active:continue
            directories=[staging/(bg+"_"+mode)/"batches"/f"{i:03d}_{i+10:03d}" for i in range(0,500,10)]
            reports=[json.loads((p/"report.json").read_text()) for p in directories]
            if any(r.get("status")!="complete" or r.get("native_step_count")!=154 or r.get("plan_identity")!=plan["plan_identity"] for r in reports):
                raise RuntimeError("Cannot publish an incomplete source-selective arm")
            target=ROOT/cell["output"];target.parent.mkdir(parents=True,exist_ok=True)
            pending=target.with_name("."+target.name+".publishing");pending.mkdir(exist_ok=False)
            products=("annual_summary.nc","mechanism_native_steps.nc","final_state.nc","parameters.nc")
            for name in products:
                merge_stream([p/name for p in directories],pending/name,plan["source_config"],
                    dict(run_identity=reports[0]["run_identity"],original_member_count=500))
            report=dict(schema=SCHEMA,status="passed_birth_source_response",scenario=bg,selector=mode,
                arm=ARM_NAMES[mode],members=500,source_config=plan["source_config"],
                run_identity=reports[0]["run_identity"],plan_identity=plan["plan_identity"],
                history_run_identity=plan["history_run_identity"],configuration=plan["solver"],
                source=cell,source_versions=plan["source_versions"],native_step_count=154,batches=reports,
                completed_at_utc=now(),products={n:dict(size_bytes=(pending/n).stat().st_size) for n in products})
            write_json(pending/"report.json",report);write_json(pending/"status.json",report)
            pending.replace(target);shutil.rmtree(staging/(bg+"_"+mode))
        analysis=output/"analysis"
        subprocess.run([sys.executable,"-B",str(ROOT/"workflows/analyze_birth_source_age_response.py"),
                        "--plan",str(args.plan),"--output",str(analysis)],cwd=ROOT,check=True)
        for name in plan["source_versions"]:shutil.copyfile(ROOT/"workflows"/name,output/(Path(name).stem+"_source_used.py"))
        execution.update(status="complete",completed_at_utc=now(),completed_member_trajectories=2000,
            completed_arms=[dict(scenario=c["scenario"],selector=c["selector"]) for c in plan["cells"]],
            reused_baselines=plan["baselines"],analysis=local(analysis),
            staging_policy="Successful duplicate batches removed; distinct scientific arms retained")
        write_json(receipt,execution);shutil.rmtree(staging)
    except BaseException as error:
        execution.update(status="failed",error=type(error).__name__+": "+str(error));write_json(receipt,execution)
        raise
    finally:lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument("--prepare",action="store_true")
    a.add_argument("--interface-check",action="store_true",help="Real multiplier interface only; no model integration")
    a.add_argument("--run",action="store_true")
    a.add_argument("--worker",action="store_true")
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--plan",type=Path)
    p.add_argument("--jobs",type=int,default=200)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--scenario",choices=SCENARIOS)
    p.add_argument("--selector",choices=SELECTORS)
    p.add_argument("--member-start",type=int,default=0)
    p.add_argument("--member-stop",type=int,default=10)
    p.add_argument("--batch-output",type=Path)
    args=p.parse_args();args.output_root=args.output_root.resolve();local(args.output_root)
    args.plan=(args.plan or args.output_root/"PLAN.json").resolve()
    if args.prepare:prepare(args)
    elif args.interface_check:interface_check(args)
    elif args.worker:
        if args.scenario is None or args.selector is None or args.batch_output is None or args.member_stop-args.member_start!=10 or args.member_start not in range(0,500,10):
            p.error("Each worker retains one original consecutive batch of ten")
        args.batch_output=args.batch_output.resolve();local(args.batch_output)
        try:worker(args)
        except BaseException as error:
            path=args.batch_output/"status.json"
            if path.exists():
                r=json.loads(path.read_text());r.update(status="failed",error=type(error).__name__+": "+str(error));write_json(path,r)
            raise
    else:
        if not 1<=args.jobs<=200:p.error("Use 1--200 single-thread workers with unchanged batches of ten")
        director(args)


if __name__=="__main__":main()

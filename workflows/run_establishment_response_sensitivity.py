"""Prepare/run one future-cohort F-response intervention in matched VL and LN.

Existing-cohort gW/gF and future-cohort gW retain their original values. Only
future-cohort gF becomes one. Reuse the completed 00, 10 and 11 arms. No core
equation, age clock, inventory, driver or event rule is changed. Realized
standing-harvest area/delivery may respond to the changed coupled state.
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

SCHEMA="matched-cmip7-establishment-response-sensitivity-v1"
SCENARIOS=("VL","LN")
SELECTORS=("new",)
ARM_NAMES={"new":"future_W_only"}
BASE_PLAN=ROOT/"outputs/age_module/birth_source_age_response/PLAN.json"
DEFAULT_OUTPUT=ROOT/"outputs/age_module/establishment_response_sensitivity"
CUTOFF=2022.


class FutureFUnmodulatedProduction:
    """Retain the original evaluator; override only actual post-2022 F slots."""
    def __init__(self,kernel,births):
        self.kernel=kernel
        self.region_id=kernel.region_id
        self.post2022=np.asarray(births)>CUTOFF

    def factors(self,age,*,member_ids):
        original=self.kernel.factors(age,member_ids=member_ids)
        if original.ndim!=3 or original.shape[1]!=2 or original.shape[-1]!=len(self.post2022):
            raise ValueError("Expected original member-by-W/F-by-birth multipliers")
        result=original.copy()
        result[:,1,self.post2022]=1.
        return result


def selected_production(kernels,births,selector):
    if selector!="new" or kernels is None:
        raise ValueError("Only the future-F-unmodulated intervention is new")
    return tuple(FutureFUnmodulatedProduction(k,births) for k in kernels)


def source_versions():
    names=("run_establishment_response_sensitivity.py","analyze_establishment_response_sensitivity.py",
           "run_birth_source_age_response.py","analyze_birth_source_age_response.py",
           "run_future_age_response.py","run_age_peak_mechanism_diagnostics.py",
           "prepare_regional_forest_age_inputs.py","run_regional_forest_age_future.py",
           "analyze_future_age_response.py")
    return {name:digest(ROOT/"workflows"/name) for name in names}


def core_source_versions():
    names=("land_carbon.py","growth_curve.py","regional_forest_age.py",
           "regional_forest_age_reference.py")
    root=ROOT/"oscar/_core/_model/_modules"
    return {name:digest(root/name) for name in names}


def prepare(args):
    base=json.loads(BASE_PLAN.read_text())
    execution=json.loads((ROOT/base["output_root"]/"execution_manifest.json").read_text())
    if base.get("schema")!="matched-cmip7-birth-source-age-response-v1" or execution.get("status")!="complete":
        raise RuntimeError("The original matched birth-source comparison must already be complete")
    ids=base["source_config"]
    if len(ids)!=500 or len(set(ids))!=500:
        raise RuntimeError("Original ordered 500 are required")
    baselines=[];cells=[]
    for bg in SCENARIOS:
        original=next(c for c in base["cells"] if c["scenario"]==bg and c["selector"]=="10")
        original_base=next(b for b in base["baselines"] if b["scenario"]==bg)
        paths=dict(original_base["paths"]);paths["10"]=original["output"]
        reports={}
        for mode,path in paths.items():
            report=json.loads((ROOT/path/"report.json").read_text())
            required={"00":"passed_future_age_response","10":"passed_birth_source_response","11":"passed_future"}[mode]
            if report.get("status")!=required or report.get("members")!=500:
                raise RuntimeError("Incomplete reusable baseline: "+bg+"/"+mode)
            if report.get("history_run_identity",report.get("configuration",{}).get("history_run_identity"))!=base["history_run_identity"]:
                raise RuntimeError("Reusable baseline has a different common history")
            with xr.open_dataset(ROOT/path/"annual_summary.nc") as annual:
                if annual.source_config.values.tolist()!=ids:
                    raise RuntimeError("Reusable baseline changed original ID order")
            if mode in original_base["run_identities"] and report["run_identity"]!=original_base["run_identities"][mode]:
                raise RuntimeError("Original 00/11 identity differs from its source plan")
            if mode=="10" and report.get("plan_identity")!=base["plan_identity"]:
                raise RuntimeError("Original existing-only baseline changed identity")
            reports[mode]=report
        baselines.append(dict(scenario=bg,paths=paths,
            run_identities={m:r["run_identity"] for m,r in reports.items()}))
        cells.append(dict(scenario=bg,selector="new",arm=ARM_NAMES["new"],
            output=local(args.output_root/bg/ARM_NAMES["new"]),
            forcing=original["forcing"],forcing_sha256=original["forcing_sha256"],
            requests=original["requests"],requests_sha256=original["requests_sha256"],
            inherited_land_driver_treatment=original["inherited_land_driver_treatment"]))
    plan=dict(schema=SCHEMA,comparison_family="Matched VL/LN; future W original and future F unmodulated",
        scenarios=list(SCENARIOS),selectors=list(SELECTORS),members_per_arm=500,source_config=ids,
        ordered_source_ids_sha256=identity(ids),anchor_year=2022,end_year=2099,
        history_directory=base["history_directory"],history_run_identity=base["history_run_identity"],
        parent_plan_identity=base["plan_identity"],output_root=local(args.output_root),
        solver=base["solver"],birth_source_cutoff=CUTOFF,new_scientific_arms=2,
        selector_definition={"new":"Existing birth_time<=2022: original gW/gF; future birth_time>2022: original gW, gF=1",
            "00":"Reuse original full production-off; age and all states continue",
            "10":"Reuse existing-cohort production response; future gW/gF=1",
            "11":"Reuse original full H1 age production"},
        fixed_inputs="Complete gross land-use/harvest drivers, event rules and establishment timing",
        endogenous_responses="Inventory, environment, supply, standing worked area and achieved harvest carbon remain coupled; not fixed realized activities",
        primary_rule="Unaged primary remains unmodulated; no age or carbon reassignment",
        interpretation="Finite conditional intervention, not independent NPP validation or a complete structural-uncertainty range",
        baselines=baselines,cells=cells,source_versions=source_versions(),core_source_versions=core_source_versions())
    plan["plan_identity"]=identity(plan);plan["prepared_at_utc"]=now()
    path=args.output_root/"PLAN.json"
    if path.exists():
        if json.loads(path.read_text()).get("plan_identity")!=plan["plan_identity"]:
            raise FileExistsError("Refusing to replace a different sensitivity plan")
        print(json.dumps(dict(status="already_prepared_not_run",plan=local(path),plan_identity=plan["plan_identity"])))
        return
    write_json(path,plan)
    print(json.dumps(dict(status="prepared_not_run",plan=local(path),plan_identity=plan["plan_identity"],
        new_scientific_arms=2,member_trajectories=1000)))


def load_plan(path):
    p=json.loads(path.read_text())
    if p.get("schema")!=SCHEMA or p.get("scenarios")!=list(SCENARIOS) or p.get("selectors")!=list(SELECTORS):
        raise RuntimeError("Not the two-pathway future-F sensitivity plan")
    if p.get("source_versions")!=source_versions() or p.get("core_source_versions")!=core_source_versions():
        raise RuntimeError("Prepared implementation changed; do not silently run a different source")
    if p.get("members_per_arm")!=500 or len(p["source_config"])!=500:
        raise RuntimeError("Original ordered 500 are required")
    return p


def interface_check(args):
    """Check the narrow multiplier interface; execute no model interval."""
    from workflows import run_regional_forest_age_future as future
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    _,layout,kernels,_=load_runtime_inputs(future.INPUTS,births=future.BIRTHS,
        tau_years=future.TAU_YEARS,age_enabled=True,member_slice=slice(0,10))
    selected=selected_production(kernels,layout.births,"new")
    for time_point in (2022.25,2050.25):
        ages=np.maximum(time_point-np.asarray(layout.births),0.)
        for original,wrapped in zip(kernels,selected):
            raw=original.factors(ages,member_ids=layout.parameter_ids)
            actual=wrapped.factors(ages,member_ids=layout.parameter_ids)
            post=wrapped.post2022
            if (not np.array_equal(actual[:,0,:],raw[:,0,:])
                or not np.array_equal(actual[:,1,~post],raw[:,1,~post])
                or not np.all(actual[:,1,post]==1.) or wrapped.region_id!=original.region_id):
                raise RuntimeError("Intervention changed W/existing F or failed to set future F to one")
    record=dict(schema=SCHEMA,status="interface_passed_no_model_integration",
        regions=len(layout.regions),members=np.asarray(layout.parameter_ids).tolist(),
        birth_slots=len(layout.births),source_cutoff=CUTOFF,solver_intervals_executed=0,
        original_W_preserved=True,existing_F_preserved=True,future_F_exactly_one=True,
        source_versions=source_versions())
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
    future_groups=[g for g in direct.forest_group.values if str(g).startswith("post2022")]
    if (np.any(direct.direct_NPP_PgC_yr.sel(tissue="F",forest_group=future_groups).values!=0.)
        or np.any(direct.direct_NPP_PgC_yr.sel(forest_group="primary_unaged").values!=0.)):
        raise RuntimeError("The consumed future-F/primary direct adjustment must be zero")
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
        masked_direct_disabled_sources=["post2022_F","primary_unaged"],sampling_reconstruction=sampling,
        diagnostics=budget,solved_atmosphere=solved,elapsed_seconds=time.perf_counter()-started)
    write_json(out/"report.json",future.plain(report));write_json(out/"status.json",future.plain(report))


def director(args):
    plan=load_plan(args.plan)
    output=ROOT/plan["output_root"]
    staging=ROOT/".runtime/establishment_response_sensitivity"/plan["plan_identity"][:16]
    staging.mkdir(parents=True,exist_ok=args.resume)
    lock=output/".run.lock";fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=output/"execution_manifest.json"
    execution=dict(schema=SCHEMA,status="running",started_at_utc=now(),plan_identity=plan["plan_identity"],
        jobs=args.jobs,batch_size=10,new_scientific_arms=2,completed_batches=[],completed_member_trajectories=0)
    try:
        active=[]
        for cell in plan["cells"]:
            target=ROOT/cell["output"]
            if target.exists():
                r=json.loads((target/"report.json").read_text())
                if not args.resume or r.get("status")!="passed_establishment_response_sensitivity" or r.get("plan_identity")!=plan["plan_identity"]:
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
            if result.returncode:raise RuntimeError("Failed establishment-sensitivity batch; see "+str(log))
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
                raise RuntimeError("Cannot publish an incomplete establishment-sensitivity arm")
            target=ROOT/cell["output"];target.parent.mkdir(parents=True,exist_ok=True)
            pending=target.with_name("."+target.name+".publishing");pending.mkdir(exist_ok=False)
            products=("annual_summary.nc","mechanism_native_steps.nc","final_state.nc","parameters.nc")
            for name in products:
                merge_stream([p/name for p in directories],pending/name,plan["source_config"],
                    dict(run_identity=reports[0]["run_identity"],original_member_count=500))
            report=dict(schema=SCHEMA,status="passed_establishment_response_sensitivity",scenario=bg,selector=mode,
                arm=ARM_NAMES[mode],members=500,source_config=plan["source_config"],
                run_identity=reports[0]["run_identity"],plan_identity=plan["plan_identity"],
                history_run_identity=plan["history_run_identity"],configuration=plan["solver"],
                source=cell,source_versions=plan["source_versions"],native_step_count=154,batches=reports,
                completed_at_utc=now(),products={n:dict(size_bytes=(pending/n).stat().st_size) for n in products})
            write_json(pending/"report.json",report);write_json(pending/"status.json",report)
            pending.replace(target);shutil.rmtree(staging/(bg+"_"+mode))
        analysis=output/"analysis"
        subprocess.run([sys.executable,"-B",str(ROOT/"workflows/analyze_establishment_response_sensitivity.py"),
                        "--plan",str(args.plan),"--output",str(analysis)],cwd=ROOT,check=True)
        for name in plan["source_versions"]:shutil.copyfile(ROOT/"workflows"/name,output/(Path(name).stem+"_source_used.py"))
        execution.update(status="complete",completed_at_utc=now(),completed_member_trajectories=1000,
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
    p.add_argument("--jobs",type=int,default=100)
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
        if not 1<=args.jobs<=100:p.error("Use 1--100 single-thread workers with unchanged batches of ten")
        director(args)


if __name__=="__main__":main()

"""Eight matched VL/LN futures: young-parcel cancellation and explicit land-service limits."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import xarray as xr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_future_age_response import write_json, identity, digest, local
from workflows.run_main_age_diagnostics import load_batch
from workflows.netcdf_batches import merge_batch_chunks
from workflows.run_age_peak_mechanism_diagnostics import netcdf
from workflows.land_supply_limited_young_protection import SupplyLimitedProtection
from workflows.land_supply_limited_retention_area_paths import area_paths

OUT = ROOT / "outputs/age_module/land_supply_limited_young_retention"
BASE = ROOT / "outputs/age_module/future_age_response/PLAN.json"
SCHEMA = "land-supply-limited-young-secondary-protection-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def versions():
    names = ["workflows/run_land_supply_limited_young_retention.py", "workflows/nontransferred_young_protection.py",
             "workflows/land_supply_limited_young_protection.py",
             "workflows/run_main_age_diagnostics.py", "workflows/main_age_process_support.py",
             "workflows/land_supply_limited_retention_area_paths.py",
             "workflows/netcdf_batches.py",
             "oscar/_core/_model/_modules/regional_forest_age.py"]
    return {n: digest(ROOT/n) for n in names}


def prepare():
    if (OUT/"PLAN.json").exists():
        raise FileExistsError("A scientific retention PLAN already exists")
    comp = json.loads((ROOT/"outputs/age_module/temperature_compensation/execution_manifest.json").read_text())
    if comp.get("status") != "complete" or len(comp.get("published_cells", [])) != 8:
        raise RuntimeError("Complete the preceding compensation experiment first")
    base = json.loads(BASE.read_text()); cells = []
    for bg in ("VL", "LN"):
        c = next(x for x in base["cells"] if x["scenario"] == bg)
        r = json.loads((ROOT/c["comparison_on"]/"report.json").read_text())
        if r.get("status") != "passed_future" or r.get("members") != 500:
            raise RuntimeError("The original complete matched on baseline is required")
        for cutoff in (10, 20, 30, 40):
            cells.append(dict(cell=f"{bg}_protect_under_{cutoff}", scenario=bg,
                cutoff_years=cutoff, production="on", forcing=c["forcing"], requests=c["requests"],
                baseline_on=c["comparison_on"], baseline_run_identity=r["run_identity"],
                output=local(OUT/bg/f"protect_under_{cutoff}")))
    p = dict(schema=SCHEMA, status="prepared_not_run", cells=cells,
        source_config=base["source_config"], ordered_source_ids_sha256=identity(base["source_config"]),
        members_per_cell=500, planned_new_scientific_configurations=8,
        history_directory=base["history_directory"], history_run_identity=base["history_run_identity"],
        base_plan=local(BASE), base_plan_identity=base["plan_identity"],
        solver=base["solver"], anchor_year=2022, end_year=2099, batch_size=10,
        source_versions=versions(), strategy="supply_limited_requests_then_young_parcel_cancellation_no_refill",
        scope="Actual age<cutoff in secdf conversion to every destination and secondary standing harvest; no invented primary age; no replacement of cancelled parcels",
        proposal_order="Original joint conversions first; original standing Q proposal on their reserved opening stocks; then cancel both protected portions",
        services="All original same-source nonforest gross requests share alpha=min(1,opening_area/sum_original_requests); conversions and that source harvest pooled; includes reduced forest establishment; actual land and wood services differ from baseline",
        land_constraint="Original requests remain unchanged as reference; nonforest actual activities have explicit uniform supply limitation with unmet service by source/destination; primary forest activities never increase; a remaining source constraint is retained by original ID",
        land_service_boundary="Nonforest requests are limited by opening land supply before protected secondary parcels are cancelled; land and wood delivery are measured outcomes",
        output="Ordinary annual climate/carbon, complete final state/parameters, annual detailed age and activity accounting including original/actual/unmet nonforest gross events and Q; no detailed native-step age cube",
        output_root=local(OUT))
    p["plan_identity"] = identity(p); p["prepared_at_utc"] = now()
    OUT.mkdir(parents=True, exist_ok=True); write_json(OUT/"PLAN.json", p)
    print(json.dumps(dict(status="prepared_not_run", plan_identity=p["plan_identity"], cells=8)))


def load_plan(path):
    p = json.loads(path.read_text())
    if p.get("schema") != SCHEMA or len(p["cells"]) != 8 or p["members_per_cell"] != 500:
        raise RuntimeError("Expected the VL/LN four-cutoff plan")
    if p["source_versions"] != versions():
        raise RuntimeError("Prepared implementation identity changed")
    return p


def worker(args):
    from workflows.run_regional_forest_age import SUMMARIES, diagnostics
    from workflows import run_regional_forest_age_future as future
    from workflows.main_age_process_support import DetailedAgeObserver
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_secondary_carbon import SEC_EXCHANGES, PROJECTION
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    plan=load_plan(args.plan); cell=next(c for c in plan["cells"] if c["cell"]==args.cell)
    selection=slice(args.member_start,args.member_stop); args.batch_output.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",cell=cell["cell"],plan_identity=plan["plan_identity"],started_at_utc=now())
    write_json(args.batch_output/"status.json",report); start=time.perf_counter()
    try:
        state,p,forcing,layout,kernels=load_batch(plan,cell,selection,age_on=True)
        policy=SupplyLimitedProtection(layout,cell["cutoff_years"]); observer=DetailedAgeObserver(layout)
        model=assemble_native_forest_age_model(layout,age_production=kernels,age_clock=None,
            secondary_agb_events=future.SECONDARY_HARVEST,harvest_projection=PROJECTION)
        keep=list(dict.fromkeys(list(SUMMARIES)+list(AGE_EXCHANGES)+list(HARVEST_EXCHANGES)+list(SEC_EXCHANGES)
            +["D_CO2","D_CH4","D_Tg","D_Td","D_Pg","D_Tl","r_Pl","D_Ebb_CH4"]))
        try:
            policy.install(); observer.install()
            annual,final=model(state,p,forcing,dtype=np.float64,var_keep=keep,
                keep_prog=False,get_final=True,nt=2,adapt_nt=False,scheme="imex")
            _,detailed=observer.datasets()
        finally:
            observer.restore(); policy.restore()
        if int(final.year)!=2099: raise RuntimeError("Incomplete policy trajectory")
        budget=diagnostics(state,final,annual); solved=future.assert_solved_atmosphere(annual,forcing)
        annual=annual.sel(year=slice(2023,2099)); annual["D_Eff"]=forcing.D_Eff.sel(year=annual.year)
        annual["D_Fland_net"]=annual.D_Fland-annual.D_Eluc
        annual.D_Fland_net.attrs.update(units="PgC yr-1",definition="D_Fland minus D_Eluc; complete native land, excluding permafrost")
        detailed=xr.merge([detailed,policy.dataset()],join="exact",compat="no_conflicts")
        run_id=identity(dict(plan_identity=plan["plan_identity"],cell=cell["cell"]))
        for ds in (annual,final,p,detailed):
            ds.attrs.update(schema=SCHEMA,run_identity=run_id,plan_identity=plan["plan_identity"],
                history_run_identity=plan["history_run_identity"],scenario=cell["scenario"],
                production="on",cutoff_years=cell["cutoff_years"],baseline_on=cell["baseline_on"])
        for ds,name in ((annual,"annual_summary.nc"),(final,"final_state.nc"),(p,"parameters.nc"),(detailed,"detailed_age_annual.nc")):
            netcdf(ds,args.batch_output/name)
        report.update(status="complete",members=10,source_config=plan["source_config"][selection],
            run_identity=run_id,diagnostics=budget,solved_atmosphere=solved,elapsed_seconds=time.perf_counter()-start,
            completed_at_utc=now())
        write_json(args.batch_output/"report.json",future.plain(report));write_json(args.batch_output/"status.json",future.plain(report))
    except BaseException as error:
        report.update(status="failed",error=type(error).__name__+": "+str(error));write_json(args.batch_output/"status.json",report);raise


def publish(job):
    plan,cell,stage=job; directories=[stage/cell["cell"]/f"{i:03d}_{i+10:03d}" for i in range(0,500,10)]
    reports=[json.loads((d/"report.json").read_text()) for d in directories]
    if any(r["status"]!="complete" or r["plan_identity"]!=plan["plan_identity"] for r in reports): raise RuntimeError("Incomplete policy members")
    target=ROOT/cell["output"];target.parent.mkdir(parents=True,exist_ok=True)
    pending=target.with_name("."+target.name+".publishing");pending.mkdir(exist_ok=False)
    for name in ("annual_summary.nc","final_state.nc","parameters.nc","detailed_age_annual.nc"):
        merge_batch_chunks([d/name for d in directories],pending/name,plan["source_config"],dict(original_member_count=500))
    report=dict(status="passed_supply_limited_young_protection",schema=SCHEMA,members=500,
        source_config=plan["source_config"],plan_identity=plan["plan_identity"],run_identity=reports[0]["run_identity"],
        configuration=cell,solver=plan["solver"],history_run_identity=plan["history_run_identity"],batches=reports,completed_at_utc=now())
    write_json(pending/"report.json",report);write_json(pending/"status.json",report);pending.replace(target)
    shutil.rmtree(stage/cell["cell"]);return cell["cell"]


def run(args):
    plan=load_plan(args.plan); stage=ROOT/".runtime/land_supply_limited_young_retention"/plan["plan_identity"][:16]
    fd=os.open(OUT/".run.lock",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644);os.write(fd,str(os.getpid()).encode());os.close(fd)
    receipt=OUT/"execution_manifest.json"
    ex=dict(schema=SCHEMA,status="area_consequences",pid=os.getpid(),plan_identity=plan["plan_identity"],
        started_at_utc=now(),jobs=args.jobs,batch_size=10,completed_batches=[],failed_batches=[],active_batches=[],published_cells=[],completed_member_trajectories=0)
    active={};mutex=threading.Lock()
    try:
        write_json(receipt,ex); areas=area_paths(plan,ROOT,OUT)
        cells=[c for c in plan["cells"] if next(x for x in areas if x["cell"]==c["cell"])["area_feasible"]]
        ex.update(status="running" if cells else "infeasible_no_coupled_cells",
                  area_infeasible_cells=[x["cell"] for x in areas if not x["area_feasible"]])
        write_json(receipt,ex)
        if not cells:
            ex.update(completed_at_utc=now(),completed_scientific_configurations=0,
                      coupled_not_run=True);write_json(receipt,ex);return
        stage.mkdir(parents=True,exist_ok=False);(stage/"logs").mkdir()
        def launch(job):
            c,i=job;key=c["cell"]+f"_{i:03d}";out=stage/c["cell"]/f"{i:03d}_{i+10:03d}";log=stage/"logs"/(key+".log")
            command=[sys.executable,"-u","-B",str(Path(__file__).resolve()),"--worker","--plan",str(args.plan),
                "--cell",c["cell"],"--member-start",str(i),"--member-stop",str(i+10),"--batch-output",str(out)]
            with log.open("w") as stream:
                p=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED="0",PYTHONPATH=str(ROOT)))
                with mutex:active[key]=dict(cell=c["cell"],first=i,pid=p.pid)
                code=p.wait()
            with mutex:active.pop(key,None)
            return dict(cell=c["cell"],first=i,returncode=code,log=local(log))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending={pool.submit(launch,(c,i)) for i in range(0,500,10) for c in cells}
            while pending:
                done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for f in done:
                    r=f.result();ex["failed_batches" if r["returncode"] else "completed_batches"].append(r)
                if ex["failed_batches"]:
                    ex["status"]="failed_workers_finishing"
                    for f in list(pending):
                        if f.cancel():pending.remove(f)
                with mutex:ex["active_batches"]=list(active.values())
                ex.update(heartbeat_at_utc=now(),completed_member_trajectories=10*len(ex["completed_batches"]));write_json(receipt,ex)
        if ex["failed_batches"]:raise RuntimeError("A policy batch failed; no automatic retry/fallback")
        ex.update(status="publishing",active_batches=[]);write_json(receipt,ex)
        with ProcessPoolExecutor(max_workers=4) as pool:
            pending={pool.submit(publish,(plan,c,stage)) for c in cells}
            while pending:
                done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
                for f in done:ex["published_cells"].append(f.result())
                ex["heartbeat_at_utc"]=now();write_json(receipt,ex)
        (stage/"logs").replace(ROOT/".runtime/land_supply_limited_young_retention/batch_logs");stage.rmdir()
        ex.update(status="complete" if len(cells)==8 else "complete_with_area_infeasible_cells",staging_removed=True,
                  completed_scientific_configurations=len(cells),completed_at_utc=now());write_json(receipt,ex)
    except BaseException as error:
        ex.update(status="failed",error=type(error).__name__+": "+str(error),failed_at_utc=now());write_json(receipt,ex);raise
    finally:(OUT/".run.lock").unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group(required=True)
    a.add_argument("--prepare",action="store_true");a.add_argument("--run",action="store_true");a.add_argument("--worker",action="store_true")
    p.add_argument("--plan",type=Path,default=OUT/"PLAN.json");p.add_argument("--jobs",type=int,default=400)
    p.add_argument("--cell");p.add_argument("--member-start",type=int,default=0);p.add_argument("--member-stop",type=int,default=10);p.add_argument("--batch-output",type=Path)
    args=p.parse_args();args.plan=args.plan.resolve()
    if args.prepare:prepare()
    elif args.worker:
        if args.member_start not in range(0,500,10) or args.member_stop!=args.member_start+10 or args.batch_output is None:p.error("Retain an original consecutive batch of ten")
        args.batch_output=args.batch_output.resolve();local(args.batch_output);worker(args)
    else:
        if not 1<=args.jobs<=500:p.error("At most 500 single-thread workers")
        run(args)


if __name__=="__main__":main()

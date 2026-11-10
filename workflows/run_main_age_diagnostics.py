"""Record detailed age diagnostics for the five full H1 futures.

Original annual variables and complete final states must match before
the additional diagnostic files are stored alongside each trajectory.
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

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import xarray as xr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.run_future_age_response import write_json, merge_stream, identity
from workflows.run_age_peak_mechanism_diagnostics import netcdf, canonical_serialized_coordinate
SCENARIOS = ("H", "HL", "M", "VL", "LN")
OUT = ROOT / "outputs/age_module/main_figure_redesign/diagnostic_replay"
STAGE = ROOT / ".runtime/main_figure_redesign/diagnostic_replay"


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    from workflows.run_future_age_response import DEFAULT_OUTPUT
    source = json.loads((DEFAULT_OUTPUT / "PLAN.json").read_text())
    cells = []
    for c in source["cells"]:
        run = ROOT / c["comparison_on"]
        report = json.loads((run / "report.json").read_text())
        if report.get("status") != "passed_future" or report.get("members") != 500:
            raise RuntimeError("A canonical H1 future is not complete: " + str(run))
        cells.append(dict(scenario=c["scenario"], output=c["comparison_on"],
                          run_identity=report["run_identity"], forcing=c["forcing"], requests=c["requests"]))
    p = dict(schema="main-age-diagnostic-replay-v1", status="prepared_not_run",
        history_directory=source["history_directory"], history_run_identity=source["history_run_identity"],
        source_config=source["source_config"], members=500, cells=cells,
        solver=source["solver"], scientific_configurations_added=0,
        age_edges_years=[0,5,10,20,30,40,60,100,"infinity"],
        new_products=["detailed_age_native_steps.nc", "detailed_age_annual.nc"],
        promotion="Only new diagnostic products are published in original H1 future directories; legacy fields and final states remain untouched")
    p["plan_identity"] = identity(p)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "PLAN.json"
    if path.exists() and json.loads(path.read_text())["plan_identity"] != p["plan_identity"]:
        raise RuntimeError("A different detailed-age diagnostic plan already exists")
    write_json(path, p)
    return p


def compare_legacy(actual, reference, selection, years=None):
    """Bounded member comparisons; output-only observers may not alter physics."""
    result = {}
    with xr.open_dataset(reference) as source:
        expected = source.isel(config=selection, missing_dims="ignore")
        if years is not None:
            expected = expected.sel(year=years)
        for name in expected.coords:
            if name not in actual.coords:
                raise RuntimeError("Missing original coordinate " + name)
            canonical_serialized_coordinate(actual[name], expected[name])
        for name in expected.data_vars:
            if name not in actual:
                raise RuntimeError("Missing original variable " + name)
            e = expected[name].load()
            a = actual[name].transpose(*e.dims)
            exact = a.dtype == e.dtype and np.array_equal(a.values, e.values, equal_nan=True)
            result[name] = bool(exact)
            if not exact:
                raise RuntimeError("Diagnostic replay changed original field " + name)
    return result


def load_batch(plan, cell, selection, age_on=True, end=2099):
    from workflows import run_regional_forest_age_future as future
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    from workflows.run_regional_forest_age import drop_prescribed_co2
    history = ROOT / plan["history_directory"]
    with xr.open_dataset(history / "final_state.nc") as ds:
        state = ds.isel(config=selection).load()
    with xr.open_dataset(history / "parameters.nc") as ds:
        parameters = ds.isel(config=selection).load()
    with xr.open_dataset(ROOT / cell["forcing"]) as ds:
        forcing = ds.isel(config=selection, missing_dims="ignore").load()
    with xr.open_dataset(ROOT / cell["requests"]) as ds:
        requests = ds[["Q_sec_harvest", "Q_sec_harvest_by_label"]].isel(config=selection, missing_dims="ignore").load()
    if "Q_sec_harvest" not in forcing:
        forcing = xr.merge([forcing, requests], join="exact", compat="no_conflicts")
    forcing = future.slice_forcing(forcing, slice(None), end)
    future.check_restart_anchor(state, forcing, "emissions")
    forcing = drop_prescribed_co2(forcing)
    ids = plan["source_config"][selection]
    for value in (state, parameters, forcing):
        if value.source_config.values.tolist() != ids:
            raise RuntimeError("Original 500-member identity/order changed")
    _, layout, kernels, _ = load_runtime_inputs(future.INPUTS, births=future.BIRTHS,
        tau_years=future.TAU_YEARS, age_enabled=age_on, member_slice=selection)
    layout.validate(state, 2022.)
    return state, parameters, forcing, layout, kernels


def integrate(state, parameters, forcing, layout, kernels, preference=None):
    from workflows.main_age_process_support import DetailedAgeObserver
    from workflows.run_regional_forest_age import SUMMARIES
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_secondary_carbon import SEC_EXCHANGES, PROJECTION
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    model = assemble_native_forest_age_model(layout, age_production=kernels, age_clock=None,
        preference=preference, secondary_agb_events=("secmf_harv", "secyf_harv"), harvest_projection=PROJECTION)
    observer = DetailedAgeObserver(layout)
    retained = list(dict.fromkeys(list(SUMMARIES)+list(AGE_EXCHANGES)+list(HARVEST_EXCHANGES)+list(SEC_EXCHANGES)
        +["D_CO2", "D_CH4", "D_Tg", "D_Td", "D_Pg", "D_Tl", "r_Pl", "D_Ebb_CH4"]))
    try:
        observer.install()
        annual, final = model(state, parameters, forcing, dtype=np.float64, var_keep=retained,
            keep_prog=False, get_final=True, nt=2, adapt_nt=False, scheme="imex")
        native, detailed = observer.datasets()
    finally:
        observer.restore()
    annual = annual.sel(year=slice(2023, 2099))
    annual["D_Eff"] = forcing.D_Eff.sel(year=annual.year)
    annual["D_Fland_net"] = annual.D_Fland - annual.D_Eluc
    annual.D_Fland_net.attrs.update(units="PgC yr-1", definition="D_Fland minus D_Eluc; all native land pools, excluding permafrost")
    return annual, final, native, detailed


def worker(args):
    plan = json.loads((OUT / "PLAN.json").read_text())
    cell = next(c for c in plan["cells"] if c["scenario"] == args.scenario)
    selection = slice(args.member_start, args.member_stop)
    path = args.batch_output
    path.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="running", scenario=args.scenario, source_config=plan["source_config"][selection],
                   plan_identity=plan["plan_identity"], started_at_utc=now())
    write_json(path / "status.json", receipt)
    start = time.perf_counter()
    try:
        state, p, forcing, layout, kernels = load_batch(plan, cell, selection)
        annual, final, native, detailed = integrate(state, p, forcing, layout, kernels)
        original = ROOT / cell["output"]
        legacy = compare_legacy(annual, original / "annual_summary.nc", selection, annual.year)
        complete_state = compare_legacy(final, original / "final_state.nc", selection)
        for ds in (native, detailed):
            ds.attrs.update(scientific_run_identity=cell["run_identity"], diagnostic_plan_identity=plan["plan_identity"], scenario=args.scenario)
        netcdf(native, path / "detailed_age_native_steps.nc")
        netcdf(detailed, path / "detailed_age_annual.nc")
        receipt.update(status="complete", completed_at_utc=now(), members=len(layout.members), native_steps=154,
            legacy_fields_unchanged=len(legacy), final_fields_unchanged=len(complete_state), elapsed_seconds=time.perf_counter()-start)
        write_json(path / "status.json", receipt)
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__+": "+str(error))
        write_json(path / "status.json", receipt)
        raise


def run(args):
    plan = prepare()
    STAGE.mkdir(parents=True, exist_ok=False)
    lock = OUT / ".run.lock"
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o644)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    receipt = dict(status="running", started_at_utc=now(), pid=os.getpid(), jobs=args.jobs,
        plan_identity=plan["plan_identity"], completed_batches=[], promoted_scenarios=[], completed_member_trajectories=0)
    write_json(OUT / "execution_manifest.json", receipt)
    def launch(job):
        bg, first = job
        out = STAGE / bg / f"{first:03d}_{first+10:03d}"
        log = STAGE / "logs" / f"{bg}_{first:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable,"-u","-B",str(Path(__file__).resolve()),"--worker","--scenario",bg,
            "--member-start",str(first),"--member-stop",str(first+10),"--batch-output",str(out)]
        with log.open("w") as stream:
            result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONHASHSEED="0",PYTHONPATH=str(ROOT)))
        if result.returncode:
            raise RuntimeError("Diagnostic batch failed: "+str(log.relative_to(ROOT)))
        return bg, first
    try:
        jobs = [(bg,i) for i in range(0,500,10) for bg in SCENARIOS]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            pending = [pool.submit(launch,j) for j in jobs]
            for f in as_completed(pending):
                bg, first = f.result()
                receipt["completed_batches"].append(dict(scenario=bg,first=first))
                receipt["completed_member_trajectories"] = 10*len(receipt["completed_batches"])
                write_json(OUT / "execution_manifest.json", receipt)
        receipt["status"] = "publishing"
        write_json(OUT / "execution_manifest.json", receipt)
        for cell in plan["cells"]:
            bg = cell["scenario"]
            target = ROOT / cell["output"]
            reports = [json.loads((STAGE/bg/f"{i:03d}_{i+10:03d}"/"status.json").read_text()) for i in range(0,500,10)]
            if any(r["status"] != "complete" or r["members"] != 10 for r in reports):
                raise RuntimeError("Incomplete original population cannot be promoted")
            promote_lock = target / ".detailed_age_promotion.lock"
            fd = os.open(promote_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644); os.close(fd)
            try:
                for name in plan["new_products"]:
                    temp = target / ("."+name+".publishing")
                    merge_stream([STAGE/bg/f"{i:03d}_{i+10:03d}"/name for i in range(0,500,10)],
                        temp,plan["source_config"],dict(scientific_run_identity=cell["run_identity"], original_member_count=500))
                    temp.replace(target/name)
                write_json(target/"detailed_age_diagnostics.json",dict(status="complete", members=500, native_steps=154,
                    scientific_run_identity=cell["run_identity"], diagnostic_plan_identity=plan["plan_identity"],
                    completed_at_utc=now(), products=plan["new_products"], legacy_products="Unmodified; all legacy annual and final-state fields compared per batch", batches=reports))
            finally:
                promote_lock.unlink(missing_ok=True)
            receipt["promoted_scenarios"].append(bg)
            write_json(OUT/"execution_manifest.json",receipt)
            shutil.rmtree(STAGE/bg)
        shutil.rmtree(STAGE)
        receipt.update(status="complete",completed_at_utc=now(), completed_member_trajectories=2500, staging_removed=True)
        write_json(OUT/"execution_manifest.json",receipt)
    except BaseException as error:
        receipt.update(status="failed",error=type(error).__name__+": "+str(error))
        write_json(OUT/"execution_manifest.json",receipt)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true");g.add_argument("--run",action="store_true");g.add_argument("--worker",action="store_true")
    p.add_argument("--jobs",type=int,default=250);p.add_argument("--scenario",choices=SCENARIOS)
    p.add_argument("--member-start",type=int,default=0);p.add_argument("--member-stop",type=int,default=10)
    p.add_argument("--batch-output",type=Path)
    a=p.parse_args()
    if a.prepare: print(json.dumps(prepare()))
    elif a.worker: worker(a)
    else:
        if not 1<=a.jobs<=250: p.error("Detailed-age diagnostics use at most 250 single-thread workers")
        run(a)

if __name__=="__main__": main()

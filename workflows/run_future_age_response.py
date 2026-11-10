"""Five complete CMIP7 futures: production off from the same H1-2022 state.

Existing full-pathway H1-on futures are the comparison, not restoration arms.
This creates five new scientific arms. It never overwrites an existing arm.
Passive process observers are reused from the already prepared diagnostic pair.
Preparation is metadata-only; --run is explicit and requires that pair complete.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "full-cmip7-common-h1-future-age-response-v1"
SCENARIOS = ("H", "HL", "M", "VL", "LN")
ARM = "off_from_H1_2022"
DEFAULT_OUTPUT = ROOT / "outputs/age_module/future_age_response"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All sources and outputs must remain inside the study")
    return str(path.relative_to(ROOT))


def source_versions():
    names = ["run_future_age_response.py", "run_age_peak_mechanism_diagnostics.py",
             "analyze_future_age_response.py"]
    return {name: digest(ROOT / "workflows" / name) for name in names}


def prepare(args):
    from workflows import run_regional_forest_age_future as future
    history = future.history_report("H1", "emissions")
    history_dir = future.history_directory("H1", "emissions")
    with xr.open_dataset(history_dir / "parameters.nc") as parameters:
        ids = parameters.source_config.values.tolist()
        positions = parameters.config.values.tolist()
    if len(ids) != 500 or len(set(ids)) != 500 or positions != list(range(500)):
        raise RuntimeError("H1 history does not retain the original ordered 500")
    cells = []
    for scenario in SCENARIOS:
        on = ROOT / "outputs/age_module/future_emissions" / scenario / "H1"
        report = json.loads((on / "report.json").read_text())
        configuration = report["configuration"]
        if report.get("status") != "passed_future" or report.get("members") != 500:
            raise RuntimeError("The existing full-pathway H1 future is incomplete: " + scenario)
        required = dict(scenario=scenario, age_production="on", atmosphere="emissions",
                        nt=2, dtype="float64", end_year=2099, batch_size=10,
                        history_policy="area_proportional",
                        harvest_contract="secondary_gross_agb_request_proportional",
                        history_run_identity=history["run_identity"])
        if any(configuration.get(k) != v for k, v in required.items()) or float(configuration["tau_years"]) != 2.0:
            raise RuntimeError("The existing on arm has a different continuation contract: " + scenario)
        with xr.open_dataset(on / "annual_summary.nc") as annual:
            if annual.source_config.values.tolist() != ids or not np.array_equal(annual.year.values, np.arange(2022, 2100)):
                raise RuntimeError("The existing on arm has a different population/time axis")
        # Reuse the exact forcing actually supplied to the existing on arm,
        # including its already declared supply treatment, not a fresh cap.
        forcing = Path(configuration["forcing"])
        if not forcing.is_absolute():
            forcing = ROOT / forcing
        requests = future.request_path(scenario)
        cells.append(dict(scenario=scenario, age_production="off", arm=ARM,
            output=local(args.output_root / scenario / ARM),
            comparison_on=local(on), comparison_on_identity=report["run_identity"],
            forcing=local(forcing), forcing_sha256=digest(forcing),
            requests=local(requests), requests_sha256=digest(requests),
            inherited_land_driver_treatment=report.get("harvest_supply_guard", {})))
    payload = dict(schema=SCHEMA, comparison_family="Full, matched CMIP7 pathways; unretimed land activity",
        scenarios=list(SCENARIOS), members_per_arm=500, source_config=ids,
        ordered_source_ids_sha256=identity(ids), history_directory=local(history_dir),
        history_run_identity=history["run_identity"], anchor_year=2022, end_year=2099,
        solver=dict(nt=2, scheme="imex", adapt_nt=False, dtype="float64", hashseed=0),
        production_reference="gW=gF=1 is host production without age modulation; age, stocks, turnover and harvest continue",
        output_root=local(args.output_root), cells=cells, source_versions=source_versions(),
        scope="Only future production response differs, from a common full H1 emissions-2022 state; not historical H0 and not controlled restoration backgrounds")
    payload["plan_identity"] = identity(payload)
    payload["prepared_at_utc"] = now()
    path = args.output_root / "PLAN.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("plan_identity") != payload["plan_identity"]:
            raise FileExistsError("Refusing to replace a different prepared scientific plan")
        print(json.dumps({"status": "already_prepared", "plan": local(path), "plan_identity": payload["plan_identity"]}))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    print(json.dumps({"status": "prepared_not_run", "plan": local(path), "plan_identity": payload["plan_identity"],
                      "new_scientific_arms": 5, "member_trajectories": 2500}))


def load_plan(path):
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA or plan.get("scenarios") != list(SCENARIOS):
        raise RuntimeError("Not the five-pathway common-H1 future-age plan")
    if plan.get("source_versions") != source_versions():
        raise RuntimeError("Prepared workflow identity changed; do not silently run a different implementation")
    if plan.get("members_per_arm") != 500 or len(set(plan["source_config"])) != 500:
        raise RuntimeError("A plan must preserve the original 500")
    return plan


def add_production(annual, direct):
    direct = direct.rename({"tissue": "mechanism_tissue", "forest_group": "mechanism_forest_group",
                            "origin": "mechanism_origin", "age_support": "mechanism_group_age_support"})
    for target, source in [("mechanism_direct_NPP", "direct_NPP_PgC_yr"),
                           ("mechanism_actual_NPP_by_origin", "actual_NPP_PgC_yr")]:
        annual[target] = direct[source]
        annual[target].attrs.update(units="PgC yr-1", diagnostic_schema=SCHEMA,
            meaning="q*A*(g-1) at the realized state" if "direct" in target else "Actual W/F NPP by birth origin and native midpoint age support")


def validate_sampling(native, annual):
    """Finite publication checks for newly saved actual integration inputs."""
    dt = native.right.values - native.left.values
    if len(dt) != 154 or not np.array_equal(native.left.values, np.arange(2022, 2099, .5)) or not np.all(dt == .5):
        raise RuntimeError("Native diagnostic sampling is not the original nt=2 support")
    errors = {}
    for short, variable in [("Tg", "D_Tg"), ("Td", "D_Td"), ("CO2", "D_CO2")]:
        before = native[short].values
        actual_after = np.vstack([before[1:], native["final_" + short].values[None]])
        reconstructed = before + dt[:, None] * native["d_" + short].values / (1 + dt[:, None] * native["vLin_" + variable].values[None])
        errors[short + "_native_max_abs"] = float(np.max(np.abs(actual_after - reconstructed)))
        means = actual_after.reshape(77, 2, -1).mean(axis=1)
        errors[short + "_annual_max_abs"] = float(np.max(np.abs(means - annual[variable].transpose("year", "config").values)))
    signs = np.array([1, 1, 1, -1, -1, 1, 1])
    supplied = (native.CO2_budget_input.values * signs).sum(axis=-1) / native.parameter_a_CO2.values[None]
    errors["CO2_derivative_max_abs_ppm_per_year"] = float(np.max(np.abs(supplied - native.d_CO2.values)))
    errors["effective_ERF_sum_max_abs_W_m2"] = float(np.max(np.abs(native.effective_ERF.values.sum(axis=-1) - native.ERF.values)))
    # These are numerical recording/reconstruction tolerances, not scientific
    # effect thresholds. No extra scientific run is opened for these checks.
    if max(errors.values()) > 1e-9:
        raise RuntimeError("New process output did not retain the actual native sampling: " + str(errors))
    return errors


def worker(args):
    from workflows import run_regional_forest_age_future as future
    from workflows.run_age_peak_mechanism_diagnostics import Observer, netcdf
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    from workflows.run_regional_forest_age import SUMMARIES, diagnostics, drop_prescribed_co2
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_secondary_carbon import SEC_EXCHANGES, PROJECTION
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    plan = load_plan(args.plan)
    cell = next(c for c in plan["cells"] if c["scenario"] == args.scenario)
    selection = slice(args.member_start, args.member_stop)
    ids = plan["source_config"][selection]
    history_dir = ROOT / plan["history_directory"]
    history = future.history_report("H1", "emissions")
    if history["run_identity"] != plan["history_run_identity"]:
        raise RuntimeError("The common H1 history changed")
    with xr.open_dataset(history_dir / "final_state.nc") as source:
        state = source.isel(config=selection).load()
    with xr.open_dataset(history_dir / "parameters.nc") as source:
        parameters = source.isel(config=selection).load()
    with xr.open_dataset(ROOT / cell["forcing"]) as source:
        forcing = source.isel(config=selection, missing_dims="ignore").load()
    with xr.open_dataset(ROOT / cell["requests"]) as source:
        requests = source[["Q_sec_harvest", "Q_sec_harvest_by_label"]].isel(config=selection, missing_dims="ignore").load()
    if "Q_sec_harvest" not in forcing:
        forcing = xr.merge([forcing, requests], join="exact", compat="no_conflicts")
    for value in (state, parameters, forcing):
        if value.source_config.values.tolist() != ids:
            raise RuntimeError("A source changed the original ordered parameter members")
    forcing = future.slice_forcing(forcing, slice(None), 2099)
    future.check_restart_anchor(state, forcing, "emissions")
    forcing = drop_prescribed_co2(forcing)
    _, layout, kernels, _ = load_runtime_inputs(future.INPUTS, births=future.BIRTHS,
        tau_years=future.TAU_YEARS, age_enabled=False, member_slice=selection)
    if kernels is not None or not np.array_equal(state.birth_time.values, np.asarray(layout.births)):
        raise RuntimeError("This arm must retain the H1 state/birth slots and only disable production modulation")
    layout.validate(state, 2022.)
    model = assemble_native_forest_age_model(layout, age_production=None, age_clock=None,
        secondary_agb_events=future.SECONDARY_HARVEST, harvest_projection=PROJECTION)
    out = args.batch_output
    out.mkdir(parents=True, exist_ok=False)
    run_id = identity(dict(plan_identity=plan["plan_identity"], scenario=args.scenario, arm=ARM))
    report = dict(schema=SCHEMA, status="running", run_identity=run_id,
        plan_identity=plan["plan_identity"], scenario=args.scenario, arm=ARM,
        member_start=args.member_start, member_stop=args.member_stop,
        source_config=ids, history_run_identity=history["run_identity"], source_versions=source_versions())
    write_json(out / "status.json", report)
    observer = Observer(model, layout, parameters, 2099)
    started = time.perf_counter()
    try:
        observer.install()
        retained = list(dict.fromkeys(list(SUMMARIES) + list(AGE_EXCHANGES) + list(HARVEST_EXCHANGES) + list(SEC_EXCHANGES) + ["D_CO2", "D_CH4", "D_Tg", "D_Td", "D_Pg", "D_Tl", "r_Pl", "D_Ebb_CH4"]))
        annual_all, final = model(state, parameters, forcing, dtype=np.float64,
            var_keep=retained, keep_prog=False, get_final=True, nt=2, adapt_nt=False, scheme="imex")
        native, direct = observer.datasets(final)
    finally:
        observer.restore()
    if int(final.year) != 2099:
        raise RuntimeError("Future did not reach 2099")
    budget = diagnostics(state, final, annual_all)
    solved = future.assert_solved_atmosphere(annual_all, forcing)
    annual = annual_all.sel(year=slice(2023, 2099))
    annual["D_Eff"] = forcing.D_Eff.sel(year=annual.year)
    annual["D_Fland_net"] = annual.D_Fland - annual.D_Eluc
    annual.D_Fland_net.attrs.update(units="PgC yr-1", definition="D_Fland minus D_Eluc; all native land classes/pools, excluding permafrost")
    if np.any(direct.direct_NPP_PgC_yr.values != 0):
        raise RuntimeError("An off arm recorded a nonzero direct production multiplier effect")
    sampling = validate_sampling(native, annual)
    add_production(annual, direct)
    attrs = dict(run_identity=run_id, family=SCHEMA, scenario=args.scenario,
        arm=ARM, age_production="off", history_run_identity=history["run_identity"],
        plan_identity=plan["plan_identity"], comparison_on=cell["comparison_on"],
        comparison_on_identity=cell["comparison_on_identity"], history_policy="area_proportional",
        harvest_contract="secondary_gross_agb_request_proportional", harvest_projection=PROJECTION,
        time_convention="Annual duration-weighted native means 2023-2099; common H1 historical annual2022 analysis anchor")
    for dataset in (annual, native, final, parameters):
        dataset.attrs.update(attrs)
    for dataset, name in [(annual, "annual_summary.nc"), (native, "mechanism_native_steps.nc"),
                          (final, "final_state.nc"), (parameters, "parameters.nc")]:
        netcdf(dataset, out / name)
    report.update(status="complete", members=len(ids), native_step_count=len(observer.records),
        sampling_reconstruction=sampling, diagnostics=budget, solved_atmosphere=solved,
        elapsed_seconds=time.perf_counter() - started)
    write_json(out / "report.json", future.plain(report))
    write_json(out / "status.json", future.plain(report))


def merge_stream(paths, output, ids, attrs):
    """Concatenate original config slices without loading full cohort states."""
    from netCDF4 import Dataset
    with Dataset(paths[0]) as template, Dataset(output, "w", format="NETCDF4") as destination:
        for name, dim in template.dimensions.items():
            destination.createDimension(name, 500 if name == "config" else len(dim))
        destination.setncatts({**{k: template.getncattr(k) for k in template.ncattrs()}, **attrs})
        for name, variable in template.variables.items():
            options = {}
            if "_FillValue" in variable.ncattrs():
                options["fill_value"] = variable.getncattr("_FillValue")
            if variable.dtype is not str and np.dtype(variable.dtype).kind not in ("O", "S", "U"):
                options.update(zlib=True, complevel=2)
            target = destination.createVariable(name, variable.datatype, variable.dimensions, **options)
            target.setncatts({k: variable.getncattr(k) for k in variable.ncattrs() if k != "_FillValue"})
        position = 0
        for path in paths:
            with Dataset(path) as chunk:
                count = len(chunk.dimensions["config"])
                if chunk.variables["source_config"][:].tolist() != ids[position:position + count]:
                    raise RuntimeError("Batch identity/order changed during publication")
                for name, variable in chunk.variables.items():
                    target = destination.variables[name]
                    if "config" in variable.dimensions:
                        index = [slice(None)] * variable.ndim
                        index[variable.dimensions.index("config")] = slice(position, position + count)
                        target[tuple(index)] = variable[:]
                    elif position == 0:
                        target[...] = variable[...]
                    elif not np.array_equal(target[...], variable[...], equal_nan=np.dtype(variable.dtype).kind in ("f", "c") if variable.dtype is not str else False):
                        raise RuntimeError("A non-member coordinate/quantity changed between batches: " + name)
                position += count
        if position != 500:
            raise RuntimeError("An arm did not assemble all original 500")


def director(args):
    plan = load_plan(args.plan)
    output_root = ROOT / plan["output_root"]
    staging = ROOT / ".runtime/future_age_response" / plan["plan_identity"][:16]
    staging.mkdir(parents=True, exist_ok=args.resume)
    lock = output_root / ".run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    execution = dict(schema=SCHEMA, status="running", started_at_utc=now(),
        plan_identity=plan["plan_identity"], jobs=args.jobs, batch_size=10,
        completed_batches=[], completed_member_trajectories=0, new_scientific_arms=5)
    receipt = output_root / "execution_manifest.json"
    try:
        active = []
        for cell in plan["cells"]:
            target = ROOT / cell["output"]
            if target.exists():
                report = json.loads((target / "report.json").read_text())
                if not args.resume or report.get("status") != "passed_future_age_response" or report.get("plan_identity") != plan["plan_identity"]:
                    raise FileExistsError("An existing scientific arm cannot be overwritten: " + str(target))
            else:
                active.append(cell["scenario"])
        write_json(receipt, execution)
        jobs = [(b, i, i + 10) for i in range(0, 500, 10) for b in active]

        def launch(job):
            scenario, first, last = job
            out = staging / scenario / "batches" / f"{first:03d}_{last:03d}"
            if out.exists():
                previous = json.loads((out / "status.json").read_text())
                if not args.resume or previous.get("plan_identity") != plan["plan_identity"]:
                    raise RuntimeError("An unknown staging batch already exists")
                if previous.get("status") == "complete":
                    return job
                shutil.rmtree(out)
            log = staging / "launch" / f"{scenario}_{first:03d}_{last:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                "--worker", "--plan", str(args.plan), "--scenario", scenario,
                "--member-start", str(first), "--member-stop", str(last), "--batch-output", str(out)]
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT)))
            if result.returncode:
                raise RuntimeError("Failed future batch; see " + str(log))
            return job

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(launch, job) for job in jobs]
            for result in as_completed(futures):
                b, first, last = result.result()
                execution["completed_batches"].append(dict(scenario=b, first=first, last=last))
                write_json(receipt, execution)
        for cell in plan["cells"]:
            if cell["scenario"] not in active:
                continue
            scenario = cell["scenario"]
            directories = [staging / scenario / "batches" / f"{i:03d}_{i+10:03d}" for i in range(0, 500, 10)]
            reports = [json.loads((p / "report.json").read_text()) for p in directories]
            if any(r.get("status") != "complete" or r.get("native_step_count") != 154 or r.get("plan_identity") != plan["plan_identity"] for r in reports):
                raise RuntimeError("An incomplete or mismatched batch cannot be published")
            target = ROOT / cell["output"]
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name("." + target.name + ".publishing")
            pending.mkdir(exist_ok=False)
            attrs = dict(run_identity=reports[0]["run_identity"], original_member_count=500)
            products = ("annual_summary.nc", "mechanism_native_steps.nc", "final_state.nc", "parameters.nc")
            for name in products:
                merge_stream([p / name for p in directories], pending / name, plan["source_config"], attrs)
            report = dict(schema=SCHEMA, status="passed_future_age_response", scenario=scenario,
                arm=ARM, age_production="off", members=500, source_config=plan["source_config"],
                run_identity=reports[0]["run_identity"], plan_identity=plan["plan_identity"],
                history_run_identity=plan["history_run_identity"], comparison_on=cell["comparison_on"],
                comparison_on_identity=cell["comparison_on_identity"], configuration=plan["solver"],
                source=cell, source_versions=plan["source_versions"], batches=reports,
                native_step_count=154, scope=plan["scope"], completed_at_utc=now(),
                products={name: dict(size_bytes=(pending/name).stat().st_size) for name in products})
            write_json(pending / "report.json", report)
            write_json(pending / "status.json", report)
            pending.replace(target)
            shutil.rmtree(staging / scenario)
        analysis = output_root / "analysis"
        subprocess.run([sys.executable, "-B", str(ROOT / "workflows/analyze_future_age_response.py"),
                        "--plan", str(args.plan), "--output", str(analysis)], cwd=ROOT, check=True)
        for name in plan["source_versions"]:
            shutil.copyfile(ROOT / "workflows" / name, output_root / (Path(name).stem + "_source_used.py"))
        execution.update(status="complete", completed_at_utc=now(), completed_member_trajectories=2500,
            completed_arms=list(SCENARIOS), analysis=local(analysis),
            staging_policy="Only new scientific arms remain; successful per-batch duplicate staging removed")
        write_json(receipt, execution)
        shutil.rmtree(staging)
    except BaseException as error:
        execution.update(status="failed", error=type(error).__name__ + ": " + str(error))
        write_json(receipt, execution)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Write the five-arm plan; start no model")
    action.add_argument("--run", action="store_true", help="Run the prepared five-arm common-H1 experiment")
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--jobs", type=int, default=250, help="Total single-thread workers across five pathways, at most 250")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--member-start", type=int, default=0)
    parser.add_argument("--member-stop", type=int, default=10)
    parser.add_argument("--batch-output", type=Path)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    local(args.output_root)
    args.plan = (args.plan or args.output_root / "PLAN.json").resolve()
    if args.prepare:
        prepare(args)
    elif args.worker:
        if not args.scenario or args.batch_output is None or args.member_stop - args.member_start != 10 or args.member_start not in range(0, 500, 10):
            parser.error("Workers retain one original consecutive batch of ten")
        args.batch_output = args.batch_output.resolve()
        local(args.batch_output)
        try:
            worker(args)
        except BaseException as error:
            path = args.batch_output / "status.json"
            if path.exists():
                record = json.loads(path.read_text())
                record.update(status="failed", error=type(error).__name__ + ": " + str(error))
                write_json(path, record)
            raise
    else:
        if not 1 <= args.jobs <= 250:
            parser.error("Use 1--250 single-thread workers; original batch size remains ten")
        director(args)


if __name__ == "__main__":
    main()

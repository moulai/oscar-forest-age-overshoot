"""Passive diagnostics for the existing VL 2030/5 production-on/off pair.

All scientific functions, forcing and states are the formal restoration ones.
Process-local observers record only values consumed by actual derivative calls.
No shared model source, original result or campaign configuration is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "age-peak-mechanism-diagnostics-v1"
GROUPS = ("existing_0_to_5", "existing_5_to_100", "existing_over_100",
          "post2022_0_to_5", "post2022_5_to_100", "primary_unaged")
COMPONENTS = ("CO2", "other_wmghg", "stratospheric_water", "ozone",
              "aerosol", "snow_particles", "land_use", "aviation_cloud",
              "volcanic_effective", "solar")
BUDGET = ("D_Eff", "D_Eluc", "D_Foxi_CH4", "D_Focean", "D_Fland",
          "D_Epf_CO2", "D_Emiss_CO2")

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def netcdf(ds, path):
    ds.to_netcdf(path, engine="netcdf4", encoding={
        v: {"zlib": True, "complevel": 2} for v in ds.data_vars})


def canonical_serialized_coordinate(actual, expected):
    """Retain exact labels while resolving xarray's object-string storage form.

    Native expand_dims constructs a string Index with object dtype. NetCDF
    decodes the same labels as fixed-width Unicode. Only that representation
    difference is canonicalized, after exact dimensions, values and order match.
    Numeric coordinate dtypes and every scientific variable remain strict.
    """
    if actual.dims != expected.dims or not np.array_equal(actual.values, expected.values):
        raise RuntimeError("Changed original coordinate values/dimensions: " + str(expected.name))
    if actual.dtype == expected.dtype:
        return actual, False
    if (actual.dtype.kind == "O" and expected.dtype.kind == "U"
            and all(isinstance(value, str) for value in actual.values.flat)):
        converted = actual.astype(expected.dtype)
        if np.array_equal(converted.values, expected.values):
            # Use the original stored coordinate, including its metadata.
            return expected.copy(deep=True), True
    raise RuntimeError("Changed original coordinate dtype: " + str(expected.name))


class Observer:
    """Explicit observers for this experiment; never substitute scientific values."""

    def __init__(self, model, layout, parameters, end):
        self.model, self.layout, self.p = model, layout, parameters
        self.template = parameters.source_config.astype(float)
        self.years = np.arange(2023, end + 1)
        shape = (len(self.years), len(layout.members), len(layout.regions), 2, 6)
        self.direct, self.actual = np.zeros(shape), np.zeros(shape)
        self.frame = None
        self.records = []
        self.cache = {}
        self.originals = []

    def vector(self, value, total=False):
        if not isinstance(value, xr.DataArray):
            value = xr.DataArray(value)
        if total:
            for dim in ("reg_land", "bio_land"):
                if dim in value.dims:
                    value = value.sum(dim, min_count=1)
        return xr.broadcast(value, self.template)[0].transpose("config").values.copy()

    def observing(self):
        return self.frame is not None and not self.frame.get("complete", False)

    def wrap_eq(self, name, callback):
        process = self.model[name]
        original = process.Eq

        def equation(variables, parameters):
            result = original(variables, parameters)
            if self.observing():
                callback(variables, parameters, result)
            return result

        self.originals.append((process, "Eq", original))
        process.Eq = equation

    def wrap_derivative(self, name):
        process = self.model[name]
        original = process.DiffEq

        def derivative(variables, parameters):
            result = original(variables, parameters)
            if not self.observing():
                raise RuntimeError("Derivative was not associated with a land interval")
            tag = {"D_Tg": "Tg", "D_Td": "Td", "D_CO2": "CO2"}[name]
            if tag in self.frame:
                raise RuntimeError("Repeated actual derivative within one interval")
            self.frame[tag] = self.vector(variables[name])
            self.frame["d_" + tag] = self.vector(result)
            if tag == "Tg":
                cached = self.cache["climate"]
                if not np.array_equal(self.vector(result), cached["derivative"]):
                    raise RuntimeError("Captured forcing did not supply the consumed temperature derivative")
                self.frame.update(cached)
            if tag == "CO2":
                cached = self.cache["co2"]
                if not np.array_equal(self.vector(result), cached["derivative_CO2"]):
                    raise RuntimeError("Captured budget did not supply the consumed CO2 derivative")
                self.frame.update(cached)
            if all(x in self.frame for x in ("Tg", "Td", "CO2")):
                self.frame["complete"] = True
                self.records.append(self.frame)
            return result

        self.originals.append((process, "DiffEq", original))
        process.DiffEq = derivative

    def install(self):
        from oscar._core._model._modules import regional_forest_age_native as native
        from oscar._core._model._modules.regional_forest_age_diagnostics import age_support_index
        original = native.advance_carbon

        def advance(state, layout, left, right, rates, factors=None, cohort_ages=None):
            if self.frame is not None and not self.frame.get("complete", False):
                raise RuntimeError("Previous interval did not consume all actual derivatives")
            self.frame = {"left": left, "right": right}
            self.cache = {}
            year = int(np.ceil(right)) - 2023
            ages = age_support_index(layout.births, (left + right) / 2)
            births = np.asarray(layout.births)
            masks = [(births <= 2022) & (ages == j) for j in range(3)]
            masks += [(births > 2022) & (ages == j) for j in range(2)]
            area = state.rf_A_forest.values
            dt = right - left
            for pool in range(2):
                q = rates.production[:, :, pool]
                for group, mask in enumerate(masks):
                    a = area[..., mask]
                    # Boolean indexing separately preserves member/region/class order.
                    g = 1. if factors is None else factors[:, :, pool, :, :][..., mask]
                    self.direct[year, :, :, pool, group] += dt * q * (a * (g - 1.)).sum(axis=(2, 3))
                    self.actual[year, :, :, pool, group] += dt * q * (a * g).sum(axis=(2, 3))
                self.actual[year, :, :, pool, 5] += dt * q * state.rf_A_primary.values
            return original(state, layout, left, right, rates, factors,
                            **({} if cohort_ages is None else {"cohort_ages": cohort_ages}))

        native.advance_carbon = advance
        self.originals.append((native, "advance_carbon", original))

        def erf(v, p, result):
            names = ("ERF_CO2", "ERF_nonCO2", "ERF_swv", "ERF_O3", "ERF_aer",
                     "ERF_lap", "ERF_luc", "ERF_aic", "ERF_volc", "ERF_solar")
            values = [self.vector(v[name]) for name in names]
            values[8] *= self.vector(p.e_volc)
            self.cache["erf"] = (self.vector(result), np.stack(values, axis=-1))

        def climate(v, p, result):
            total, components = self.cache["erf"]
            if not np.array_equal(total, self.vector(v.ERF)):
                raise RuntimeError("ERF capture does not match the actual climate input")
            self.cache["climate"] = {"ERF": total, "ERF_components": components,
                                      "derivative": self.vector(result)}

        def sink(v, p, result):
            self.cache["sink"] = {k: self.vector(v[k], total=True)
                                  for k in ("D_Focean", "D_Fland", "D_Epf_CO2")}

        def co2(v, p, result):
            terms = dict(self.cache["sink"])
            terms.update({k: self.vector(v[k], total=True)
                          for k in ("D_Eff", "D_Eluc", "D_Foxi_CH4", "D_Emiss_CO2")})
            self.cache["co2"] = {"budget": np.stack([terms[k] for k in BUDGET], axis=-1),
                                  "derivative_CO2": self.vector(result)}

        self.wrap_eq("ERF", erf)
        self.wrap_eq("d_Tg", climate)
        self.wrap_eq("D_Fsink_CO2", sink)
        self.wrap_eq("d_CO2", co2)
        for name in ("D_Tg", "D_Td", "D_CO2"):
            self.wrap_derivative(name)

    def restore(self):
        for obj, attribute, original in reversed(self.originals):
            setattr(obj, attribute, original)

    def datasets(self, final):
        if not self.records or not self.records[-1].get("complete", False):
            raise RuntimeError("Incomplete native diagnostic history")
        coords = dict(config=list(self.layout.members),
                      source_config=("config", list(self.layout.parameter_ids)))
        native = xr.Dataset(coords={**coords, "step": np.arange(len(self.records)),
            "forcing_component": list(COMPONENTS), "co2_component": list(BUDGET)})
        for name in ("left", "right"):
            native[name] = ("step", [r[name] for r in self.records])
        for name in ("Tg", "Td", "CO2", "d_Tg", "d_Td", "d_CO2", "ERF"):
            native[name] = (("step", "config"), np.stack([r[name] for r in self.records]))
        native["effective_ERF"] = (("step", "config", "forcing_component"),
                                    np.stack([r["ERF_components"] for r in self.records]))
        native["CO2_budget_input"] = (("step", "config", "co2_component"),
                                      np.stack([r["budget"] for r in self.records]))
        for short, name in [("Tg", "D_Tg"), ("Td", "D_Td"), ("CO2", "D_CO2")]:
            native["final_" + short] = ("config", self.vector(final[name]))
        for name in ("Th_g", "Th_d", "lambda_0", "e_ohu", "th_0", "a_CO2"):
            native["parameter_" + name] = ("config", self.vector(self.p[name]))
        for name in ("D_Tg", "D_Td", "D_CO2"):
            native["vLin_" + name] = ("config", self.vector(self.model[name].vLin(self.p)))
        native.attrs.update(schema=SCHEMA, time_support="Actual values consumed by native derivatives; not post-output evaluations",
            forcing_support="Effective ERF terms in the exact order used by total ERF, including volcanic efficacy",
            scheme="imex", nt=2, initial_difference="Same complete H1 2022 state; paired component temperature differences start at zero")
        direct = xr.Dataset({
            "direct_NPP_PgC_yr": (("year", "config", "reg_land", "tissue", "forest_group"), self.direct),
            "actual_NPP_PgC_yr": (("year", "config", "reg_land", "tissue", "forest_group"), self.actual)},
            coords={**coords, "year": self.years, "reg_land": list(self.layout.regions),
                    "tissue": ["W", "F"], "forest_group": list(GROUPS),
                    "origin": ("forest_group", ["existing_2022"] * 3 + ["post2022"] * 2 + ["primary_unaged"]),
                    "age_support": ("forest_group", ["0_to_5", "5_to_100", "over_100", "0_to_5", "5_to_100", "primary_unaged"])})
        direct.attrs.update(schema=SCHEMA, counterfactual="q*A*(g-1) at the same realized environment, area and state",
            sampling="Sum over actual native land intervals; annual duration is one year", age_support="Native midpoint age: <5, 5..100 inclusive, >100",
            origin_rule="Actual birth_time<=2022 versus >2022; all establishment and harvest renewal, not only retimed restoration",
            primary_rule="Unaged; direct factor effect identically zero; actual production retained",
            interpretation="Production diagnostic, not age-resolved net uptake or temperature attribution")
        for name in direct:
            direct[name].attrs["units"] = "PgC yr-1"
        native.effective_ERF.attrs["units"] = "W m-2"
        native.ERF.attrs["units"] = "W m-2"
        native.CO2_budget_input.attrs.update(units="PgC yr-1", signs="+,+,+,-,-,+,+")
        for name in ("Tg", "Td", "final_Tg", "final_Td"):
            native[name].attrs.update(units="K", temperature_reference="Raw model anomaly, not rebased to 1850-1900")
        for name in ("d_Tg", "d_Td"):
            native[name].attrs["units"] = "K yr-1"
        for name in ("CO2", "final_CO2"):
            native[name].attrs["units"] = "ppm"
        native.d_CO2.attrs["units"] = "ppm yr-1"
        for name in ("left", "right"):
            native[name].attrs["units"] = "year"
        for name in ("Th_g", "Th_d", "lambda_0", "e_ohu", "th_0", "a_CO2"):
            native["parameter_" + name].attrs.update(self.p[name].attrs)
        for name in ("D_Tg", "D_Td", "D_CO2"):
            native["vLin_" + name].attrs["units"] = "yr-1"
        return native, direct


def worker(args):
    from workflows import run_restoration_emissions as r
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    from workflows.run_regional_forest_age import extend_birth_slots, SUMMARIES
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_secondary_carbon import SEC_EXCHANGES
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    from oscar._core._model._modules.regional_forest_secondary_carbon import PROJECTION
    plan = r.load_plan()
    cell = r.select_cell(plan, "VL", 2030, 5, args.age_production)
    selection = slice(args.member_start, args.member_stop)
    history, state, parameters = r.load_restart_slice(plan, selection)
    forcing = r.compose_forcing(cell, selection, solved=True).sel(year=slice(2022, args.end_year))
    _, layout, kernels, _ = load_runtime_inputs(r.INPUTS, births=r.BIRTHS,
        tau_years=r.TAU_YEARS, age_enabled=args.age_production == "on", member_slice=selection)
    state = extend_birth_slots(state, layout, year=2022)
    model = assemble_native_forest_age_model(layout, age_production=kernels, age_clock=None,
        secondary_agb_events=r.SECONDARY_HARVEST, harvest_projection=PROJECTION)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="running", mode=args.age_production,
        member_start=args.member_start, member_stop=args.member_stop,
        source_config=plan["source_config"][selection], end_year=args.end_year,
        workflow_sha256=digest(__file__), history_run_identity=history["run_identity"],
        input_manifest_sha256=plan["input_manifest_sha256"], scope="VL 2030 start / 5-year calendar span; shared H1 emissions 2022 state")
    write_json(out / "status.json", report)
    observer = Observer(model, layout, parameters, args.end_year)
    started = time.perf_counter()
    try:
        observer.install()
        retained = list(dict.fromkeys(list(SUMMARIES) + list(AGE_EXCHANGES) + list(HARVEST_EXCHANGES) + list(SEC_EXCHANGES) + ["D_CO2", "D_CH4", "D_Tg", "D_Td", "D_Pg", "D_Tl", "r_Pl", "D_Ebb_CH4"]))
        annual, final = model(state, parameters, forcing, dtype=np.float64,
            var_keep=retained, keep_prog=False, get_final=True,
            nt=2, adapt_nt=False, scheme="imex")
        native, direct = observer.datasets(final)
    finally:
        observer.restore()
    annual = annual.sel(year=slice(2023, args.end_year))
    annual["D_Eff"] = forcing.D_Eff.sel(year=annual.year)
    annual["D_Fland_net"] = annual.D_Fland - annual.D_Eluc
    reference = ROOT / cell["output"] / "annual_summary.nc"
    errors = {}
    with xr.open_dataset(reference) as source:
        for name in source.data_vars:
            expected = source[name].isel(config=selection, missing_dims="ignore").sel(year=annual.year).load()
            actual = annual[name].transpose(*expected.dims)
            delta = np.abs(actual.values - expected.values)
            exact = bool(np.array_equal(actual.values, expected.values, equal_nan=True))
            errors[name] = dict(exact=exact, same_dims=actual.dims == expected.dims,
                same_dtype=actual.dtype == expected.dtype,
                max_abs=0.0 if exact else float(np.nanmax(delta)))
        coordinate_normalizations = {}
        for name in source.coords:
            expected = source[name]
            if "config" in expected.dims:
                expected = expected.isel(config=selection)
            if "year" in expected.dims:
                expected = expected.sel(year=annual.year)
            if name not in annual.coords:
                raise RuntimeError("Changed original coordinate: " + name)
            coordinate, normalized = canonical_serialized_coordinate(annual[name], expected)
            if normalized:
                coordinate_normalizations[name] = dict(
                    memory_dtype=str(annual[name].dtype), stored_dtype=str(expected.dtype),
                    exact_values_and_order=True, reason="NetCDF string representation")
                annual = annual.assign_coords({name: coordinate})
    if any(not x["exact"] or not x["same_dims"] or not x["same_dtype"] for x in errors.values()):
        write_json(out / "trajectory_comparison.json", errors)
        raise RuntimeError("Diagnostic observer materially changed a retained formal trajectory")
    for ds in (annual, native, direct):
        ds.attrs.update(mode=args.age_production, history_run_identity=history["run_identity"],
                        input_manifest_sha256=plan["input_manifest_sha256"], original_member_count=500)
    netcdf(annual, out / "annual_summary.nc")
    netcdf(native, out / "native_mechanism.nc")
    netcdf(direct, out / "direct_production.nc")
    write_json(out / "trajectory_comparison.json", errors)
    report.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                  native_step_count=len(observer.records), trajectory_max_abs=max(x["max_abs"] for x in errors.values()),
                  all_compared_fields_exact=all(x["exact"] for x in errors.values()),
                  coordinate_serialization_normalizations=coordinate_normalizations)
    write_json(out / "report.json", report)
    write_json(out / "status.json", report)


def promote(args, execution):
    """Publish only new diagnostics; retain all original scientific payloads."""
    from workflows import run_restoration_emissions as r
    plan = r.load_plan()
    targets = {mode: ROOT / r.select_cell(plan, "VL", 2030, 5, mode)["output"]
               for mode in ("on", "off")}
    locks, prepared = [], []
    try:
        for target in targets.values():
            lock = target / ".mechanism_diagnostics.lock"
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            locks.append(lock)
        for mode, target in targets.items():
            original = xr.load_dataset(target / "annual_summary.nc")
            replay = xr.load_dataset(args.output / mode / "annual_summary.nc")
            for name in original.data_vars:
                if name.startswith("mechanism_"):
                    continue
                expected, actual = original[name], replay[name].transpose(*original[name].dims)
                if actual.dtype != expected.dtype or not np.array_equal(actual.values, expected.values, equal_nan=True):
                    raise RuntimeError("Promotion would change an original variable: " + name)
            for name in original.coords:
                if not name.startswith("mechanism_") and (original[name].dtype != replay[name].dtype or not np.array_equal(original[name], replay[name])):
                    raise RuntimeError("Promotion would change an original coordinate: " + name)
            added = xr.load_dataset(args.output / mode / "direct_production.nc").rename({
                "tissue": "mechanism_tissue", "forest_group": "mechanism_forest_group",
                "origin": "mechanism_origin", "age_support": "mechanism_group_age_support"})
            additions = {"mechanism_direct_NPP": "direct_NPP_PgC_yr",
                         "mechanism_actual_NPP_by_origin": "actual_NPP_PgC_yr"}
            for name, source in additions.items():
                original[name] = added[source]
                original[name].attrs.update(units="PgC yr-1", diagnostic_schema=SCHEMA,
                    meaning=("q*A*(g-1) at the realized environment/area/state" if "direct" in name
                             else "Actual W/F production by actual birth origin and native age support"))
            original.attrs.update(mechanism_diagnostics_schema=SCHEMA,
                                  mechanism_diagnostics_provenance="mechanism_diagnostics.json")
            annual_temp = target / ".annual_summary.mechanism.tmp.nc"
            native_temp = target / ".mechanism_native_steps.tmp.nc"
            netcdf(original, annual_temp)
            import shutil
            shutil.copyfile(args.output / mode / "native_mechanism.nc", native_temp)
            metadata = dict(schema=SCHEMA, status="complete", mode=mode,
                scientific_run_identity=original.attrs.get("run_identity"),
                history_run_identity=original.attrs.get("history_run_identity"),
                input_manifest_sha256=plan["input_manifest_sha256"],
                workflow_sha256=digest(__file__), members=500, source_config=plan["source_config"],
                scientific_configuration="Unchanged VL 2030/5 production-on/off; same H1 2022 restart, equations, parameters and solver",
                original_payload_comparison="All original annual variables and coordinates identical in value and dtype; original values and attributes retained",
                added_annual_variables=list(additions), native_file="mechanism_native_steps.nc",
                native_sampling="Actual native derivative inputs and their effective ERF components, not output-time evaluations",
                analysis=str(args.analysis_output.relative_to(ROOT)))
            prepared.append((target, annual_temp, native_temp, metadata))
        # Only fully computed and compared pairs reach this point. Each file
        # replacement is atomic; the per-run receipt is committed last.
        for target, annual_temp, native_temp, metadata in prepared:
            annual_temp.replace(target / "annual_summary.nc")
            native_temp.replace(target / "mechanism_native_steps.nc")
            metadata["files"] = {name: dict(size_bytes=(target / name).stat().st_size,
                                             sha256=digest(target / name))
                                 for name in ("annual_summary.nc", "mechanism_native_steps.nc")}
            write_json(target / "mechanism_diagnostics.json", metadata)
        execution["promoted_to_original_runs"] = {mode: str(path.relative_to(ROOT)) for mode, path in targets.items()}
        execution["staging_policy"] = "Removed after successful promotion; no duplicate annual/final-state/parameter archive"
    finally:
        for _, annual_temp, native_temp, _ in prepared:
            annual_temp.unlink(missing_ok=True)
            native_temp.unlink(missing_ok=True)
        for lock in locks:
            lock.unlink(missing_ok=True)


def director(args):
    from workflows import run_restoration_emissions as r
    plan = r.load_plan()
    if args.promote and not args.output.resolve().is_relative_to(ROOT / ".runtime/age_peak_mechanism"):
        raise ValueError("Promoted replays must use the dedicated project-internal staging directory")
    if not args.promote:
        raise ValueError("Formal replay requires --promote; short functional checks use --worker")
    if args.end_year != 2099:
        raise ValueError("A formal director run must retain the full 2023-2099 trajectory")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    signature = digest(__file__)
    jobs = [(mode, i, min(500, i + args.batch_size))
            for i in range(0, 500, args.batch_size) for mode in ("on", "off")]
    report = dict(schema=SCHEMA, status="running", members_per_arm=500,
        source_config=plan["source_config"], jobs_per_arm=args.jobs_per_arm,
        batch_size=args.batch_size, workflow_sha256=signature, completed_batches=[])
    write_json(args.output / "status.json", report)

    def launch(job):
        mode, first, last = job
        out = args.output / mode / "batches" / f"{first:03d}_{last:03d}"
        if (out / "report.json").exists() and args.resume:
            done = json.loads((out / "report.json").read_text())
            if done.get("status") == "complete" and done.get("workflow_sha256") == signature:
                return mode, first, last
            raise RuntimeError("Existing batch is not complete under this workflow identity")
        if out.exists() and args.resume:
            status = out / "status.json"
            previous = json.loads(status.read_text()) if status.exists() else {}
            if previous.get("schema") != SCHEMA or previous.get("workflow_sha256") != signature:
                raise RuntimeError("Refusing to remove an unrecognized or changed staging batch")
            import shutil
            shutil.rmtree(out)
        log = args.output / "launch" / f"{mode}_{first:03d}_{last:03d}.log"
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "--worker",
            "--age-production", mode, "--member-start", str(first), "--member-stop", str(last),
            "--output", str(out)]
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONHASHSEED="0", PYTHONPATH=str(ROOT)))
        if result.returncode:
            raise RuntimeError(f"Failed diagnostic batch {mode}/{first:03d}_{last:03d}; see {log}")
        return mode, first, last

    with ThreadPoolExecutor(max_workers=args.jobs_per_arm * 2) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for future in as_completed(futures):
            mode, first, last = future.result()
            report["completed_batches"].append(dict(mode=mode, first=first, last=last))
            write_json(args.output / "status.json", report)
    for mode in ("on", "off"):
        directories = sorted((args.output / mode / "batches").iterdir())
        for name in ("annual_summary.nc", "native_mechanism.nc", "direct_production.nc"):
            chunks = [xr.load_dataset(path / name) for path in directories]
            merged = xr.concat(chunks, dim="config", data_vars="minimal", coords="minimal", compat="equals")
            if merged.source_config.values.tolist() != plan["source_config"]:
                raise RuntimeError("Merged diagnostics changed original member identity/order")
            netcdf(merged, args.output / mode / name)
    command = [sys.executable, "-B", str(ROOT / "workflows/analyze_age_peak_mechanism_diagnostics.py"),
               "--diagnostics", str(args.output), "--output", str(args.analysis_output)]
    if args.promote:
        command.append("--published")
    subprocess.run(command, cwd=ROOT, check=True)
    report["batch_reports"] = [json.loads((args.output / mode / "batches" / f"{first:03d}_{last:03d}" / "report.json").read_text()) for mode, first, last in jobs]
    import shutil
    shutil.copyfile(Path(__file__), args.analysis_output / "diagnostic_replay_source_used.py")
    if args.promote:
        promote(args, report)
    report.update(status="complete", completed_member_trajectories=1000)
    write_json(args.analysis_output / "execution_manifest.json", report)
    if args.promote:
        import shutil
        shutil.rmtree(args.output)
    else:
        write_json(args.output / "report.json", report)
        write_json(args.output / "status.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, default=ROOT / "outputs/age_module/restoration_emissions/analysis/age_peak_mechanism_v1")
    parser.add_argument("--promote", action="store_true", help="Atomically add diagnostics to the existing formal runs, then remove staging")
    parser.add_argument("--jobs-per-arm", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--age-production", choices=["on", "off"])
    parser.add_argument("--member-start", type=int, default=0)
    parser.add_argument("--member-stop", type=int, default=500)
    parser.add_argument("--end-year", type=int, default=2099)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.analysis_output = args.analysis_output.resolve()
    try:
        if args.worker:
            if args.age_production is None or not 0 <= args.member_start < args.member_stop <= 500:
                parser.error("Worker requires an age mode and a valid original positional slice")
            worker(args)
        else:
            if not 1 <= args.jobs_per_arm <= 50 or args.batch_size != 10:
                parser.error("Formal pair uses batch size 10 and at most 50 workers per arm")
            director(args)
    except BaseException as error:
        status = args.output / "status.json"
        if status.exists():
            record = json.loads(status.read_text())
            if record.get("schema") == SCHEMA:
                record.update(status="failed", error=str(error))
                write_json(status, record)
        raise



if __name__ == "__main__":
    main()

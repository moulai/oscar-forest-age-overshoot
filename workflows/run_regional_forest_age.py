#!/usr/bin/env python3
"""Run staged native regional forest-age smoke tests and dynamic PI spin-up.

This workflow uses the ordinary Model time loop. Full cohort states are saved
only as restart products; retained interval outputs are explicit summaries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs, fingerprint

SUMMARIES = ['rf_NBP', 'rf_NPP', 'rf_Efire', 'rf_HWP_input', 'rf_budget_residual',
             'rf_ref_budget_residual', 'rf_event_count', 'D_Fland', 'D_Eluc',
             'D_Aland', 'D_Frss', 'Cveg', 'Ccwd', 'Csoil', 'Chwp', 'Aland',
             'rf_forest_W_mean', 'rf_forest_F_mean', 'rf_forest_area_mean',
             'rf_NBP_biome', 'rf_NPP_W', 'rf_NPP_F', 'rf_Eresp', 'rf_Ehwp',
             'rf_Ecwd', 'rf_Esoil', 'rf_Ecrop', 'rf_Egraz']


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stock_change(before, after):
    """Per-member actual total change; reference densities are not actual stocks."""
    from oscar._core._model._modules.regional_forest_age import total_carbon
    return total_carbon(after) - total_carbon(before)


def diagnostics(before, after, output):
    """Audit actual total carbon against its same-interval atmospheric uptake."""
    integrated = output.rf_NBP.isel(year=slice(1, None)).sum(('year', 'reg_land'))
    residual = stock_change(before, after) - integrated
    values = [v.values for n, v in after.data_vars.items() if n.startswith('rf_') and v.size]
    minimum = min(float(np.min(v)) for v in values)
    if not all(np.isfinite(v).all() for v in values) or minimum < -1e-10:
        raise ValueError('Nonfinite or materially negative physical state')
    maximum_residual = float(abs(residual).max())
    throughput = float(abs(output.rf_NPP.isel(year=slice(1, None))).sum(('year', 'reg_land', 'bio_land')).max())
    tolerance = 1e-8 + 1e-8 * max(1., throughput)
    if maximum_residual > tolerance:
        raise ValueError('Actual carbon budget exceeds the declared throughput-scaled tolerance')
    return {'stock_change_PgC': [float(stock_change(before, after).min()), float(stock_change(before, after).max())],
            'maximum_cumulative_budget_residual_PgC': maximum_residual,
            'budget_diagnostic_tolerance_PgC': tolerance, 'minimum_state': minimum,
            'maximum_reported_step_residual_rate_PgC_per_year': float(abs(output.rf_budget_residual).max()),
            'event_count': float(output.rf_event_count.isel(year=slice(1, None)).sum())}


def pi_forcing(layout, start, end):
    years = np.arange(start, end + 1)
    return xr.Dataset({'event_area': (('year', 'reg_land', 'event'), np.zeros((len(years), len(layout.regions), len(layout.events)))),
                       'D_CO2': ('year', np.zeros(len(years))), 'D_Tl': ('year', np.zeros(len(years))),
                       'r_Pl': ('year', np.ones(len(years)))},
                      coords={'year': years, 'reg_land': list(layout.regions), 'event': [e.event_id for e in layout.events]})


def density_window_change(before, after, atol, rtol):
    """Convergence is local to every member, region and reference density pool."""
    largest = 0.
    max_change = 0.
    location = {}
    for name in ('rf_ref_live', 'rf_ref_cwd', 'rf_ref_soil'):
        a, b = before[name], after[name]
        change = abs(a-b)
        normalized = change / (atol + rtol * np.maximum(abs(a), abs(b)))
        local_max = float(normalized.max())
        if local_max >= largest:
            indices = np.unravel_index(np.argmax(normalized.values), normalized.shape)
            location = {'variable': name, **{dim: str(normalized[dim].values[i]) if dim in normalized.coords else int(i)
                                               for dim, i in zip(normalized.dims, indices)}}
            if 'config' in normalized.dims and 'source_config' in after.coords:
                location['source_config'] = str(after.source_config.sel(config=normalized.config.values[indices[normalized.dims.index('config')]]).item())
        largest = max(largest, local_max)
        max_change = max(max_change, float(change.max()))
    return largest, max_change, location


def expand_legacy_area_state(state):
    """Explicit read boundary for shared geometry in legacy checkpoints."""
    result = state.copy(deep=False)
    for name in ("rf_A_primary", "rf_A_forest", "rf_A_nonforest"):
        if "config" not in result[name].dims:
            result[name] = result[name].expand_dims(config=result.config).copy(deep=True)
    return result


def extend_birth_slots(state, layout, *, year):
    """Extend empty future slots once at a phase boundary; retain all stocks.

    Existing birth slots may not be dropped or relabeled. Resetting the time
    origin after PI spin-up does not assign a birth date to the primary forest.
    """
    old = np.asarray(state.birth_time.values)
    target = np.asarray(layout.births)
    if not np.isin(old, target).all():
        raise ValueError('A stage transition cannot discard existing birth slots')
    result = state.reindex(birth_time=target, fill_value=0.).assign_coords(year=float(year))
    result = expand_legacy_area_state(result)
    layout.validate(result, float(year))
    return result


def initialize_full_native_state(model, parameters, forcing, land_state=None):
    """Use standard native initial states, then retain the complete land restart.

    Prescribed initial prognostics come from the first forcing record. This
    keeps the initialization row consistent with concentration-driven inputs.
    No climate state is inferred from a land restart.
    """
    coordinates = {} if land_state is None else {name: value for name, value in land_state.coords.items() if name != 'year'}
    template = parameters.assign_coords(coordinates)
    initial = model._get_Ini(template, forcing)
    if land_state is not None:
        land_state = expand_legacy_area_state(land_state)
        for name, value in land_state.data_vars.items():
            initial[name] = value
    for name in model.var_prog | model.var_node:
        if name in forcing and 'year' in forcing[name].dims:
            initial[name] = forcing[name].isel(year=0, drop=True)
    return initial.assign_coords(year=float(forcing.year[0]))


def compare_smoke_runs(first, second):
    """Compare same-input resolution probes in absolute carbon units."""
    first, second = Path(first), Path(second)
    a, b = [xr.load_dataset(path / 'annual_summary.nc') for path in (first, second)]
    if not np.array_equal(a.source_config, b.source_config):
        raise ValueError('Resolution probes use different original member orders')
    result = {}
    def compare(x, y):
        x, y = xr.align(x, y, join='exact')
        x, y = [v.sum([d for d in v.dims if d != 'config']) for v in (x, y)]
        difference = abs(x-y)
        relative = difference / np.maximum(abs(y), 1e-10)
        return {'first_range_PgC': [float(x.min()), float(x.max())],
                'maximum_absolute_difference_PgC': float(difference.max()),
                'maximum_relative_difference': float(relative.max()),
                'median_relative_difference': float(relative.median())}
    for name in ('rf_NBP', 'rf_NPP', 'rf_Efire', 'rf_HWP_input', 'D_Eluc'):
        result['cumulative_' + name] = compare(a[name].isel(year=slice(1, None)), b[name].isel(year=slice(1, None)))
    a, b = [xr.load_dataset(path / 'final_state.nc') for path in (first, second)]
    groups = {'forest_W': ('rf_W_primary', 'rf_W_forest'), 'forest_F': ('rf_F_primary', 'rf_F_forest'),
              'nonforest_live': ('rf_C_nonforest',), 'cwd': ('rf_C_cwd',), 'soil': ('rf_C_soil',), 'hwp': ('rf_C_hwp',)}
    for group, names in groups.items():
        values = [sum(v[name].sum([d for d in v[name].dims if d != 'config']) for name in names) for v in (a, b)]
        result['final_' + group] = compare(*values)
    result['maximum_area_difference_Mha'] = max(float(abs(a[name]-b[name]).max()) for name in ('rf_A_primary', 'rf_A_forest', 'rf_A_nonforest'))
    return result


def validate_spinup_source(source_report, input_record):
    """Reject a stable restart belonging to different scientific inputs."""
    if source_report['status'] != 'passed_dynamic_spinup':
        raise ValueError('Historical initialization requires a stable dynamic spin-up')
    if source_report['input_manifest']['sha256'] != input_record['sha256']:
        raise ValueError('Spin-up and historical input manifest identities differ')
    if source_report['configuration']['phase'] != 'spinup' or source_report['birth_slots'] != 0:
        raise ValueError('Historical initialization requires no-birth PI spin-up')



from oscar._core._model._modules.regional_forest_secondary_carbon import CANDIDATE, PROJECTION, SEC_EXCHANGES

HARVEST_CONTRACTS = ('area', CANDIDATE)
SECONDARY_EVENTS = ('secmf_harv', 'secyf_harv')


def secondary_extension_identity(directory, parent_manifest):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['parent_input_manifest_sha256'] != parent_manifest['sha256']:
        raise ValueError('Secondary extension belongs to a different prepared input bundle')
    identity = {name: fingerprint(directory / name) for name in
                ('manifest.json', 'prehistory_requests.nc', 'historical_requests.nc', 'projection_parameters.nc')}
    for name, value in identity.items():
        if name != 'manifest.json' and value['sha256'] != manifest['outputs'][name]['sha256']:
            raise ValueError('Secondary extension file differs from its manifest: ' + name)
    return identity


def load_secondary_history_extension(directory, primary, parameters, prehistory, history):
    """Attach independent extension inputs without changing original forcing."""
    projection = xr.load_dataset(directory / 'projection_parameters.nc').sel(config=primary.config)
    for coordinate in ('config','source_config','reg_land'):
        if not np.array_equal(projection[coordinate], primary[coordinate]):
            raise ValueError('Secondary projection coordinates differ: ' + coordinate)
    if projection.k_F_AG.attrs.get('projection') != PROJECTION:
        raise ValueError('Secondary projection definition is missing or unsupported')
    k, leaf = projection.k_F_AG, projection.k_F_AG_leaf_only
    if (not np.isfinite(k).all() or not np.isfinite(leaf).all() or
            bool(((leaf < 0) | (leaf > k) | (k > 1)).any())):
        raise ValueError('Projection requires 0 <= k_F_AG_leaf_only <= k_F_AG <= 1')
    parameters = xr.merge([parameters, projection[['k_F_AG','k_F_AG_leaf_only']]], join='exact', compat='no_conflicts')
    forcing = []
    for original, filename in ((prehistory,'prehistory_requests.nc'), (history,'historical_requests.nc')):
        extra = xr.load_dataset(directory / filename)
        for coordinate in ('year','reg_land'):
            if not np.array_equal(extra[coordinate], original[coordinate]):
                raise ValueError('Secondary request support differs: ' + coordinate)
        if tuple(extra.secondary_harvest_label.values) != SECONDARY_EVENTS:
            raise ValueError('Secondary request label order differs')
        for name in ('Q_sec_harvest','Q_sec_harvest_by_label'):
            if extra[name].attrs.get('units') != 'PgC':
                raise ValueError('Secondary annual requests require PgC per interval')
            if np.any(extra[name].isel(year=0).values != 0):
                raise ValueError('Secondary request first row must be a zero anchor')
        forcing.append(xr.merge([original,extra[['Q_sec_harvest','Q_sec_harvest_by_label']]], join='exact'))
    return parameters, *forcing


HISTORY_POLICIES = ('area_proportional', 'purpose_maturity', 'purpose_rank')
# Solved by the native atmosphere. Leaving either CO2 name in For prescribes it.
SOLVED_ATMOSPHERE = ('D_CO2', 'd_CO2', 'D_Tg', 'D_Td')
PRESCRIBED_CONCENTRATIONS = ('D_CH4', 'd_CH4', 'D_N2O', 'd_N2O')


def drop_prescribed_co2(forcing):
    """Integrate CO2 and temperature. Keep prescribed CH4 and N2O concentrations.

    D_Eff stays the age historical series. Prehistory is unchanged: its D_CO2 is
    the zero preindustrial land environment, not an atmospheric trajectory.
    """
    required = ('D_Eff', 'D_CO2', 'd_CO2') + PRESCRIBED_CONCENTRATIONS
    missing = [name for name in required if name not in forcing]
    if missing:
        raise ValueError('Emissions-driven history is missing drivers: ' + ', '.join(missing))
    if forcing['D_Eff'].attrs.get('units') != 'PgC yr-1':
        raise ValueError('D_Eff units are not PgC yr-1')
    if not bool(np.isfinite(forcing['D_Eff']).all()):
        raise ValueError('D_Eff contains non-finite values')
    solved = forcing.drop_vars([name for name in SOLVED_ATMOSPHERE if name in forcing])
    leaked = [name for name in SOLVED_ATMOSPHERE if name in solved]
    if leaked:
        raise ValueError('Solved atmosphere remained in the forcing: ' + ', '.join(leaked))
    if any(name not in solved for name in PRESCRIBED_CONCENTRATIONS):
        raise ValueError('CH4 or N2O concentrations were dropped')
    return solved


def history_policy(configuration):
    """Legacy reports without a policy describe proportional history only."""
    policy = configuration.get('history_policy', 'area_proportional')
    if policy not in HISTORY_POLICIES:
        raise ValueError('Unknown historical selection policy')
    return policy


def history_preference(directory, layout, policy):
    """Assemble fixed regional S for two explicitly named prepared activities.

    Historical purpose is a controlled sensitivity: mature-secondary harvest
    uses 1+S and young-secondary harvest uses 2-S. Production and stocks never
    enter S. Proportional and rank selection use no finite-weight callback.
    """
    if policy in ('area_proportional', 'purpose_rank'):
        return None
    if policy != 'purpose_maturity':
        raise ValueError('Unknown historical selection policy')
    from oscar._core._model._modules.growth_curve import load_growth_curve
    from oscar._core._model._modules.regional_forest_management import management_weights
    # Label meaning belongs to prepared input assembly, never the model core.
    managed = ('secmf_harv', 'secyf_harv')
    events = {event.event_id: event for event in layout.events}
    for name in managed:
        event = events.get(name)
        if (event is None or event.source_class != 'secdf' or event.target_class != 'secdf'
                or event.kind != 'harvest' or event.vegetation_action != 'clear'
                or event.age_action != 'reset'):
            raise ValueError('Prepared secondary purpose harvest identity/semantics differ: ' + name)
    curve = load_growth_curve(Path(directory) / 'growth_curve.nc', weighting='capacity')
    def preference(variables, parameters, forcing, event_time):
        return management_weights(layout, event_time,
            lambda ages: curve.evaluate(ages, regions=layout.regions)[0],
            mode='purpose_maturity', managed_events=managed, strength=1.)
    return preference


def history_rank_selection(layout, policy):
    """Name the strong two-ended source candidate at the experiment boundary."""
    if policy != 'purpose_rank':
        return None
    from oscar._core._model._modules.regional_forest_age import validate_purpose_rank
    directions = {'secyf_harv': 'youngest', 'secmf_harv': 'oldest'}
    if validate_purpose_rank(layout, directions) != 'secdf':
        raise ValueError('Prepared purpose_rank activities must use secdf')
    return directions


def run_history(args):
    """Continuous 850 prehistory and 1750-2022 native history from dynamic PI spin-up."""
    from oscar._core._model._modules.regional_forest_age_reference import (
        build_native_forest_age_model, assemble_native_forest_age_model,
    )
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from workflows.prepare_regional_forest_age_inputs import exact_coordinates
    if args.spinup_directory is None:
        raise ValueError('Historical runs require --spinup-directory')
    source_report = json.loads((args.spinup_directory / 'report.json').read_text())
    inputs = fingerprint(args.inputs / 'manifest.json')
    validate_spinup_source(source_report, inputs)
    member_slice = slice(args.member_start, args.member_stop)
    partial = (args.member_start, args.member_stop) != (0, 500)
    primary, layout, kernels, _ = load_runtime_inputs(args.inputs, births=np.arange(850.5, 2099., 1.),
                         tau_years=args.tau_years, age_enabled=args.age_production == 'on', member_slice=member_slice)
    policy = history_policy(vars(args))
    secondary = args.harvest_contract == CANDIDATE
    if secondary and policy != 'area_proportional':
        raise ValueError('Secondary carbon history requires proportional selection')
    if secondary != (args.secondary_carbon_inputs is not None):
        raise ValueError('Secondary contract and extension directory must be specified together')
    secondary_options = dict(secondary_agb_events=SECONDARY_EVENTS, harvest_projection=PROJECTION) if secondary else {}
    extension = secondary_extension_identity(args.secondary_carbon_inputs, inputs) if secondary else None
    preference = history_preference(args.inputs, layout, policy)
    purpose_rank = history_rank_selection(layout, policy)
    p = xr.load_dataset(args.spinup_directory / 'baseline_parameters.nc').isel(config=member_slice)
    exact_coordinates(primary, p)
    spun = xr.load_dataset(args.spinup_directory / 'final_state.nc').isel(config=member_slice)
    exact_coordinates(primary, spun)
    if any(value.attrs.get('run_identity') != source_report['run_identity'] for value in (p, spun)):
        raise ValueError('Spin-up baseline and state have inconsistent run identities')
    state = extend_birth_slots(spun, layout, year=850.)
    prehistory = xr.load_dataset(args.inputs / 'prehistory_forcing.nc')
    history = xr.load_dataset(args.inputs / 'historical_forcing.nc').isel(config=member_slice)
    if (float(prehistory.year[0]), float(prehistory.year[-1]), float(history.year[0]), float(history.year[-1])) != (850., 1750., 1750., 2022.):
        raise ValueError('Historical phase boundaries differ from the prepared contract')
    if secondary:
        p, prehistory, history = load_secondary_history_extension(args.secondary_carbon_inputs, primary, p, prehistory, history)
    if args.emissions_driven:
        history = drop_prescribed_co2(history)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['history_policy'] = policy
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    identity = hashlib.sha256(json.dumps({'configuration': config, 'inputs': inputs['sha256'],
                               'spinup': source_report['run_identity'], 'code': code, **({'secondary_extension': extension} if secondary else {})}, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True)
    checkpoints = args.output / 'checkpoints'
    checkpoints.mkdir()
    report = {'status': 'running', 'run_identity': identity, 'code_commit': code, 'configuration': config,
              'input_manifest': inputs, 'secondary_extension': extension, 'spinup_identity': source_report['run_identity'],
              'spinup_code_commit': source_report['code_commit'], 'source_config': p.source_config.values.tolist(),
              'members': len(layout.members), 'birth_slots': len(layout.births), 'phases': [],
              'scope': 'Completed native trajectory is subject to separate observational and scientific acceptance',
              'initialization': 'Accepted dynamic PI actual/reference stocks; empty annual births added once at 850; complete land restart retained at 1750'}
    json_write(args.output / 'configuration.json', config)
    json_write(args.output / 'status.json', report)
    p.attrs.update(run_identity=identity, history_policy=policy, harvest_contract=args.harvest_contract,
                           harvest_projection=PROJECTION if secondary else 'none')
    p.to_netcdf(args.output / 'parameters.nc')
    summaries = list(SUMMARIES) + list(AGE_EXCHANGES) + list(HARVEST_EXCHANGES) + (list(SEC_EXCHANGES) if secondary else [])
    start = time.perf_counter()
    def save(dataset, path):
        dataset.attrs.update(run_identity=identity, history_policy=policy, harvest_contract=args.harvest_contract,
                           harvest_projection=PROJECTION if secondary else 'none')
        dataset.to_netcdf(path)
    def integrate(model, before, forcing, retained):
        output, final = model(before, p, forcing, dtype=np.float64, var_keep=retained,
                              keep_prog=False, get_final=True, nt=args.nt, adapt_nt=False)
        check = diagnostics(before, final, output)
        record = {'start': int(forcing.year[0]), 'end': int(forcing.year[-1]),
                  'elapsed_seconds': time.perf_counter()-start, **check}
        report['phases'].append(record)
        save(final, checkpoints / ('state_' + str(int(final.year)) + '.nc'))
        json_write(args.output / 'status.json', report)
        with (args.output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        return output, final
    try:
        model = build_native_forest_age_model(layout, age_production=kernels, preference=preference,
                                              purpose_rank=purpose_rank, **secondary_options)
        first, state = integrate(model, state, prehistory.sel(year=slice(850, 860)), summaries)
        output, state = integrate(model, state, prehistory.sel(year=slice(860, 1750)), summaries)
        prehistory_output = xr.concat([first, output.isel(year=slice(1, None))], dim='year',
                                     join='exact', compat='equals', combine_attrs='override')
        save(prehistory_output, args.output / 'prehistory_summary.nc')
        del first, output, prehistory_output
        model = assemble_native_forest_age_model(layout, age_production=kernels, preference=preference,
                                              purpose_rank=purpose_rank, **secondary_options)
        state = initialize_full_native_state(model, p, history, state)
        retained = summaries + ['D_CO2', 'D_CH4', 'D_Tg', 'D_Pg', 'D_Tl', 'r_Pl', 'D_Ebb_CH4']
        pieces = []
        for first, last in ((1750, 2010), (2010, 2020), (2020, 2022)):
            output, state = integrate(model, state, history.sel(year=slice(first, last)), retained)
            save(output, args.output / ('history_' + str(first) + '_' + str(last) + '.nc'))
            pieces.append(output if not pieces else output.isel(year=slice(1, None)))
        annual = xr.concat(pieces, dim='year', join='exact', compat='equals', combine_attrs='override')
        save(annual, args.output / 'annual_summary.nc')
        save(state, args.output / 'final_state.nc')
        report.update(status='passed_historical_batch' if partial else 'passed_historical', elapsed_seconds=time.perf_counter()-start,
                      peak_resident_memory_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        json_write(args.output / 'report.json', report)
        json_write(args.output / 'status.json', report)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.perf_counter()-start)
        json_write(args.output / 'status.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'full-smoke', 'baseline-smoke', 'spinup', 'history'], required=True)
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--spinup-directory', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--start-year', type=int, default=2000)
    parser.add_argument('--member-start', type=int, default=0)
    parser.add_argument('--member-stop', type=int, default=500)
    parser.add_argument('--tau-years', type=float, required=True)
    parser.add_argument('--age-production', choices=['on', 'off'], default='on')
    parser.add_argument('--history-policy', choices=HISTORY_POLICIES, default='area_proportional',
                        help='Historical source selection: proportional, bounded maturity preference, or strong two-ended rank candidate')
    parser.add_argument('--harvest-contract', choices=HARVEST_CONTRACTS, default='area')
    parser.add_argument('--secondary-carbon-inputs', type=Path)
    parser.add_argument('--emissions-driven', action='store_true',
                        help='Integrate CO2 and temperature; do not prescribe D_CO2, d_CO2, D_Tg, or D_Td')
    parser.add_argument('--nt', type=int, default=2)
    parser.add_argument('--smoke-years', type=int, default=10)
    parser.add_argument('--window-years', type=int, default=100)
    parser.add_argument('--maximum-years', type=int, default=6000)
    parser.add_argument('--stock-atol', type=float, default=1e-7,
                        help='PI density change tolerance per window, PgC/Mha')
    parser.add_argument('--stock-rtol', type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace run output')
    if min(args.nt, args.smoke_years, args.window_years, args.maximum_years) <= 0 or args.stock_atol <= 0 or args.stock_rtol <= 0:
        raise ValueError('Positive time and diagnostic settings required')
    if not 0 <= args.member_start < args.member_stop <= 500:
        raise ValueError('Member bounds must be a contiguous part of the original 500')
    if args.phase != 'history' and (args.member_start, args.member_stop) != (0, 500):
        raise ValueError('Partial member execution is supported only for computational historical batches')
    if args.phase != 'history' and args.history_policy != 'area_proportional':
        raise ValueError('The purpose policy requires the complete history phase')
    if args.phase != 'history' and (args.harvest_contract != 'area' or args.secondary_carbon_inputs is not None):
        raise ValueError('Secondary carbon extension is supported only in the history phase')
    if args.emissions_driven and args.phase != 'history':
        raise ValueError('Emissions-driven CO2 is supported only in the history phase')
    if args.phase == 'history':
        return run_history(args)
    from oscar._core._model._modules.regional_forest_age_reference import (
        build_native_forest_age_model, assemble_native_forest_age_model, initialize_native_forest_state, install_preindustrial_baseline,
    )
    births = () if args.phase == 'spinup' else tuple(np.arange(850.5, 2099., 1.))
    p, layout, kernels, area = load_runtime_inputs(args.inputs, births=births, tau_years=args.tau_years,
                                                  age_enabled=args.age_production == 'on')
    model = build_native_forest_age_model(layout, age_production=kernels)
    state = initialize_native_forest_state(layout, area, year=0. if args.phase == 'spinup' else 850.)
    summaries = list(SUMMARIES)
    fixed_forcing = None
    if args.phase == 'full-smoke':
        if args.restart is None:
            raise ValueError('Full smoke requires an explicit numerical land checkpoint')
        state = xr.load_dataset(args.restart).assign_coords(year=float(args.start_year))
        # This is a declared transient numerical probe, not an accepted PI state.
        p = install_preindustrial_baseline(p, state, layout)
        model = assemble_native_forest_age_model(layout, age_production=kernels)
        fixed_forcing = xr.load_dataset(args.inputs / 'historical_forcing.nc').sel(year=slice(args.start_year, args.start_year + args.smoke_years))
        fixed_forcing['event_area'] = xr.zeros_like(fixed_forcing.event_area)
        state = initialize_full_native_state(model, p, fixed_forcing, state)
        summaries += ['D_CO2', 'D_CH4', 'D_Tg', 'D_Pg', 'D_Tl', 'r_Pl', 'D_Ebb_CH4']
    elif args.phase == 'baseline-smoke':
        if args.data_root is None:
            raise ValueError('Native baseline benchmark requires --data-root')
        from oscar._core._model.OSCAR import OSCAR
        model = OSCAR
        p = xr.load_dataset(args.data_root / 'data/processed/reference_background/v1/indexed_primary_parameters.nc')
        fixed_forcing = xr.load_dataset(args.data_root / 'data/processed/reference_background/v1/historical_concentration_forcing.nc').sel(year=slice(args.start_year, args.start_year + args.smoke_years))
        state = initialize_full_native_state(model, p, fixed_forcing)
        summaries = [name for name in ['D_Fland', 'D_Eluc', 'D_Aland', 'D_Frss', 'D_Efire',
                     'D_CO2', 'D_CH4', 'D_Tg', 'D_Pg', 'D_Tl', 'r_Pl', 'D_Ebb_CH4',
                     'D_Cveg_bk', 'D_Ccwd_bk', 'D_Csoil_bk', 'D_Chwp'] if name in model]
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {'status': 'running', 'configuration': config, 'members': len(layout.members),
              'birth_slots': len(layout.births), 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runner': fingerprint(__file__), 'input_manifest': fingerprint(args.inputs / 'manifest.json'),
              'retained_outputs': summaries, 'scope': 'Native regional land process; no claim of historical ecological validity from a zero-carbon smoke start'}
    report['run_identity'] = hashlib.sha256(json.dumps({'configuration': config, 'inputs': report['input_manifest']['sha256'], 'code': report['code_commit']}, sort_keys=True).encode()).hexdigest()
    json_write(args.output / 'status.json', report)
    json_write(args.output / 'configuration.json', config)
    start = time.perf_counter()
    def run(before, forcing):
        return model(before, p, forcing, dtype=np.float64, var_keep=summaries,
                     keep_prog=False, get_final=True, nt=args.nt, adapt_nt=False)
    try:
        if args.phase in ('smoke', 'full-smoke', 'baseline-smoke'):
            before = state.copy(deep=True)
            forcing = (xr.load_dataset(args.inputs / 'prehistory_forcing.nc').sel(year=slice(850, 850 + args.smoke_years))
                       if fixed_forcing is None else fixed_forcing)
            if forcing.sizes['year'] != args.smoke_years + 1:
                raise ValueError('Smoke exceeds prepared prehistory coverage')
            output, state = run(before, forcing)
            checks = diagnostics(before, state, output) if args.phase != 'baseline-smoke' else {}
            if args.phase in ('full-smoke', 'baseline-smoke'):
                checks['atmosphere_finite'] = all(bool(np.isfinite(output[name]).all()) for name in ['D_CO2', 'D_CH4', 'D_Tg', 'D_Tl', 'r_Pl'])
                if not checks['atmosphere_finite']:
                    raise ValueError('Nonfinite climate/carbon outputs')
                checks['initial_CO2_matches_For'] = bool((output.D_CO2.isel(year=0) == forcing.D_CO2.isel(year=0)).all())
            report.update(checks, status='passed_native_smoke',
                          years=args.smoke_years, scientific_status='Numerical/performance probe only: cold or transient relabeled initial conditions, not a historical experiment; native baseline land structure differs')
            output.attrs['run_identity'] = report['run_identity']
            output.to_netcdf(args.output / 'annual_summary.nc')
        else:
            windows, consecutive = [], 0
            year = 0
            while year < args.maximum_years:
                end = min(year + args.window_years, args.maximum_years)
                before = state.copy(deep=True)
                output, state = run(before, pi_forcing(layout, year, end))
                check = diagnostics(before, state, output)
                scaled, absolute, location = density_window_change(before, state, args.stock_atol, args.stock_rtol)
                consecutive = consecutive + 1 if scaled <= 1. else 0
                record = dict(year=end, maximum_normalized_density_change=scaled,
                              maximum_density_change_PgC_per_Mha=absolute, maximum_normalized_change_location=location,
                              final_global_NBP_range_PgC_per_year=[float(output.rf_NBP.isel(year=-1).sum('reg_land').min()),
                                                                float(output.rf_NBP.isel(year=-1).sum('reg_land').max())],
                              elapsed_seconds=time.perf_counter()-start, **check)
                windows.append(record)
                with (args.output / 'progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                year = end
                if consecutive >= 2:
                    break
            report.update(status='passed_dynamic_spinup' if consecutive >= 2 else 'spinup_horizon_exhausted',
                          years=year, windows=windows, consecutive_stable_windows=consecutive,
                          scientific_status='Numerical PI stabilization; ecological observations and timestep sensitivity remain separate gates')
            output.attrs['run_identity'] = report['run_identity']
            output.to_netcdf(args.output / 'final_window_summary.nc')
            if consecutive >= 2:
                baseline = install_preindustrial_baseline(p, state, layout)
                baseline.attrs['run_identity'] = report['run_identity']
                baseline.to_netcdf(args.output / 'baseline_parameters.nc')
        state.attrs['run_identity'] = report['run_identity']
        state.to_netcdf(args.output / 'final_state.nc')
        report.update(elapsed_seconds=time.perf_counter()-start,
                      peak_resident_memory_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
        json_write(args.output / 'report.json', report)
        json_write(args.output / 'status.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'years', 'elapsed_seconds', 'peak_resident_memory_MiB')}), flush=True)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.perf_counter()-start)
        json_write(args.output / 'status.json', report)
        raise


if __name__ == '__main__':
    main()

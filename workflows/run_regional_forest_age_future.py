#!/usr/bin/env python3
"""Continue one forest-age history through one CMIP7 scenario.

The 2022 forcing row is the initial label. Only 2023 onward is integrated, from
that arm's own 2022 state. Concentration runs prescribe CO2. Emissions runs
restart from the emissions history and solve CO2 and temperature. Secondary
stand harvest is unchanged: carbon leaves, and area and birth year stay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Hash seed is read at interpreter startup. Re-exec before NumPy so every
# continuation, including a direct smoke command, uses the same seed.
if os.environ.get('PYTHONHASHSEED') != '0':
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable, '-B', *sys.argv])

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENARIOS = ('H', 'HL', 'M', 'VL', 'LN')
ARMS = ('H0', 'H1')
ANCHOR_YEAR = 2022
LAST_FUTURE_YEAR = 2099
TAU_YEARS = 2.0
HISTORY_POLICY = 'area_proportional'
NT = 2
# Below the regional land already resolved by the 2022 states. Larger than
# the 1e-13 Mha H0/H1 area difference, so float tails are not treated as caps.
SUPPLY_TOLERANCE_MHA = 1e-8
SECONDARY_HARVEST = ('secmf_harv', 'secyf_harv')
INPUTS = ROOT / 'data/processed/regional_forest_age_inputs_v1'
FORCING_DIR = ROOT / 'outputs/01_prepared_data/cmip7_n2o_mass_corrected_H0_v1'
FUTURE_ROOT = ROOT / 'outputs/age_module/future_concentration'
ATMOSPHERES = ('concentration', 'emissions')
BIRTHS = np.arange(850.5, 2099., 1.)
_LOG_STREAMS = []


def json_write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def claim_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to replace an experiment directory: ' + str(path))
    path.mkdir(parents=True)
    return path


def mirror_stdio(path):
    """Send this process's console into the experiment directory."""
    stream = open(path, 'w', buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    _LOG_STREAMS.append(stream)


def require_study_oscar():
    import oscar
    path = Path(oscar.__file__).resolve()
    expected = (ROOT / 'oscar').resolve()
    origin = (ROOT / 'oscar_origin').resolve()
    if origin in path.parents or expected not in path.parents:
        raise RuntimeError('Imported ' + str(path) + '; expected the study oscar package')
    return str(path)


def history_directory(arm, atmosphere='concentration'):
    """H0 and H1 each keep their own history. Emissions never reads a concentration state."""
    if atmosphere not in ATMOSPHERES or arm not in ARMS:
        raise ValueError('Unknown atmosphere or arm')
    folder = 'historical_emissions' if atmosphere == 'emissions' else 'historical_concentration'
    return ROOT / 'outputs/age_module' / folder / arm


def history_report(arm, atmosphere='concentration'):
    path = history_directory(arm, atmosphere) / 'status.json'
    report = json.loads(path.read_text())
    if report.get('status') != 'passed_historical':
        raise RuntimeError(arm + ' ' + atmosphere + ' history is not a passed restart')
    config = report['configuration']
    expected_age = 'on' if arm == 'H1' else 'off'
    driven = bool(config.get('emissions_driven'))
    if atmosphere == 'emissions' and not driven:
        raise RuntimeError(arm + ' emissions future cannot restart from a concentration history')
    if atmosphere == 'concentration' and driven:
        raise RuntimeError(arm + ' concentration future cannot restart from an emissions history')
    if (float(config['tau_years']) != TAU_YEARS or config.get('history_policy') != HISTORY_POLICY
            or config.get('harvest_contract') != 'secondary_gross_agb_request_proportional'
            or config.get('age_production') != expected_age or int(config.get('nt', NT)) != NT):
        raise RuntimeError(arm + ' history configuration is not the continuation contract')
    return report


def load_restart(arm, member_slice, atmosphere='concentration'):
    """Published 2022 state written by the history runner.

    final_state.nc is that end state. checkpoints/state_2022.nc is the same
    save; D_CO2 is compared so a mismatched file cannot be used silently.
    """
    report = history_report(arm, atmosphere)
    directory = history_directory(arm, atmosphere)
    state_path = directory / 'final_state.nc'
    checkpoint = directory / 'checkpoints' / 'state_2022.nc'
    if not state_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('History runner did not write both 2022 restart files for ' + arm)
    state = xr.load_dataset(state_path).isel(config=member_slice)
    if 'year' not in state.coords or state.year.ndim != 0 or int(state.year) != ANCHOR_YEAR:
        raise RuntimeError('Restart is not the scalar 2022 state')
    parameters = xr.load_dataset(directory / 'parameters.nc').isel(config=member_slice)
    if not np.array_equal(parameters.source_config.values, state.source_config.values):
        raise RuntimeError('Restart parameters and state are not the same members')
    if 'k_F_AG' not in parameters or 'k_F_AG_leaf_only' not in parameters:
        raise RuntimeError('Restart parameters are missing the secondary projection')
    with xr.open_dataset(checkpoint) as saved:
        checkpoint_co2 = np.asarray(saved.D_CO2.isel(config=member_slice).values)
    if not np.array_equal(checkpoint_co2, np.asarray(state.D_CO2.values)):
        raise RuntimeError('final_state.nc and checkpoints/state_2022.nc disagree on D_CO2')
    return report, state, parameters


def secondary_opening(arm, atmosphere='concentration'):
    """Secondary-forest area at the 2022 boundary. Members share one area trajectory."""
    path = history_directory(arm, atmosphere) / 'final_state.nc'
    with xr.open_dataset(path) as state:
        area = state.rf_A_forest.sel(forest_class='secdf').sum('birth_time').load()
    spread = float((area.max('config') - area.min('config')).max())
    if spread > SUPPLY_TOLERANCE_MHA:
        raise RuntimeError(arm + ' members do not share a secondary-area trajectory')
    opening = area.isel(config=0)
    opening.attrs['member_spread_Mha'] = spread
    return opening


def _event_table(metadata):
    events = [str(value) for value in metadata.event.values]
    source = [str(value) for value in metadata.event_source_class.values]
    target = [str(value) for value in metadata.event_target_class.values]
    kind = [str(value) for value in metadata.event_kind.values]
    incoming = [name for name, tgt, how in zip(events, target, kind)
                if tgt == 'secdf' and how == 'conversion']
    outgoing = [name for name, src, how in zip(events, source, kind)
                if src == 'secdf' and how == 'conversion']
    if tuple(name for name, src, tgt, how in zip(events, source, target, kind)
             if how == 'harvest' and src == 'secdf' and tgt == 'secdf') != SECONDARY_HARVEST:
        raise RuntimeError('Secondary-forest harvest events are not secmf_harv, secyf_harv')
    return events, incoming, outgoing


def assess_secondary_supply(opening, event_area, metadata):
    """Walk 2023-2099. Cap only secdf harvest that exceeds opening plus incoming conversion.

    Conversions and primary harvest are not edited. Stand harvest does not
    change area, so the next opening is conversions plus primf_harv only.
    """
    events, incoming, outgoing = _event_table(metadata)
    ordered = event_area.transpose('year', 'reg_land', 'event')
    if [str(value) for value in ordered.event.values] != events:
        ordered = ordered.sel(event=events)
    ordered = ordered.sel(reg_land=list(opening.reg_land.values))
    years = np.asarray(ordered.year.values)
    if int(years[0]) != ANCHOR_YEAR or int(years[-1]) != LAST_FUTURE_YEAR or not np.array_equal(years, np.arange(ANCHOR_YEAR, LAST_FUTURE_YEAR + 1)):
        raise RuntimeError('Future event_area years are not the 2022-2099 anchor contract')
    values = np.array(ordered.values, dtype=np.float64, copy=True)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError('Future event_area is negative or nonfinite')
    if float(values[0].sum()) != 0.:
        raise RuntimeError('Future anchor row is not event-free')
    index = {name: i for i, name in enumerate(events)}
    harvest_at = [index[name] for name in SECONDARY_HARVEST]
    incoming_at = [index[name] for name in incoming]
    outgoing_at = [index[name] for name in outgoing]
    primf_at = index['primf_harv']
    regions = [str(value) for value in ordered.reg_land.values]
    area = np.asarray(opening.values, dtype=np.float64).copy()
    min_slack = np.inf
    min_model_slack = np.inf
    slack_at = None
    empty_cells = []
    capped = []
    minimum_area = area.copy()
    for step, year in enumerate(years):
        if step == 0:
            continue
        row = values[step]
        incoming_area = row[:, incoming_at].sum(axis=1)
        outgoing_area = row[:, outgoing_at].sum(axis=1)
        harvest = row[:, harvest_at].sum(axis=1)
        supply = area + incoming_area
        slack = supply - harvest
        model_slack = area - outgoing_area - harvest
        local = float(slack.min())
        if local < min_slack:
            at = int(np.argmin(slack))
            min_slack = local
            slack_at = {'year': int(year), 'region': regions[at], 'opening_Mha': float(area[at]),
                        'incoming_conversion_Mha': float(incoming_area[at]), 'supply_Mha': float(supply[at]),
                        'harvest_Mha': float(harvest[at]), 'slack_Mha': float(slack[at])}
        min_model_slack = min(min_model_slack, float(model_slack.min()))
        unpaid_conversion = outgoing_area - area
        if np.any(unpaid_conversion > SUPPLY_TOLERANCE_MHA):
            at = int(np.argmax(unpaid_conversion))
            raise RuntimeError(
                'Secondary conversion exceeds opening area at '
                + f'{int(year)} {regions[at]}: conversion={outgoing_area[at]:.8g} Mha, '
                + f'opening={area[at]:.8g} Mha. Conversion is not edited.')
        for region_at in np.flatnonzero(harvest > supply + SUPPLY_TOLERANCE_MHA):
            available = float(supply[region_at])
            requested = float(harvest[region_at])
            factor = 0. if available <= 0. else available / requested
            before = row[region_at, harvest_at].copy()
            row[region_at, harvest_at] *= factor
            capped.append({'year': int(year), 'region': regions[region_at],
                           'requested_Mha': requested, 'supply_Mha': available,
                           'secmf_harv_Mha': float(before[0]), 'secyf_harv_Mha': float(before[1]),
                           'scale': float(factor)})
        empty = (area <= SUPPLY_TOLERANCE_MHA) & (incoming_area <= SUPPLY_TOLERANCE_MHA) & (harvest > SUPPLY_TOLERANCE_MHA)
        for region_at in np.flatnonzero(empty):
            empty_cells.append({'year': int(year), 'region': regions[region_at],
                                'harvest_Mha': float(harvest[region_at])})
        area = area + incoming_area + row[:, primf_at] - outgoing_area
        if np.any(area < -SUPPLY_TOLERANCE_MHA):
            raise RuntimeError('Secondary area would become negative. Refusing to invent an area rule.')
        minimum_area = np.minimum(minimum_area, area)
    me = regions.index('MIDDLE_EAST')
    manifest = {
        'cap_applied': bool(capped),
        'capped_cells': len(capped),
        'empty_secondary_harvest_cells': len(empty_cells),
        'minimum_slack_Mha': None if not np.isfinite(min_slack) else float(min_slack),
        'minimum_slack_location': slack_at,
        'minimum_opening_minus_conversion_and_harvest_Mha': float(min_model_slack),
        'middle_east': {
            'opening_2022_Mha': float(np.asarray(opening.values)[me]),
            'minimum_area_Mha': float(minimum_area[me]),
            'end_area_Mha': float(area[me]),
        },
        'capped': capped,
        'empty_cells': empty_cells,
        'tolerance_Mha': SUPPLY_TOLERANCE_MHA,
        'supply_definition': 'start-of-year secdf area plus same-year conversion into secdf',
        'not_edited': 'historical forcing, conversion, primary harvest, and secondary carbon requests',
    }
    capped_area = None
    if capped:
        rebuilt = ordered.copy(data=values)
        rebuilt.attrs = dict(ordered.attrs)
        capped_area = rebuilt.transpose('year', 'reg_land', 'event')
    return manifest, capped_area


def forcing_path(scenario):
    return FORCING_DIR / f'future_{scenario.lower()}_forcing.nc'


def request_path(scenario):
    return FORCING_DIR / f'future_{scenario.lower()}_requests.nc'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assert_forcing_contract(forcing, requests):
    historical = xr.open_dataset(INPUTS / 'historical_forcing.nc')
    try:
        if set(forcing.data_vars) != set(historical.data_vars):
            raise RuntimeError('Future forcing variable set differs from historical_forcing.nc')
        for name in historical.data_vars:
            if forcing[name].dims != historical[name].dims:
                raise RuntimeError('Future forcing dimensions differ: ' + name)
            if forcing[name].attrs.get('units') != historical[name].attrs.get('units'):
                raise RuntimeError('Future forcing units differ: ' + name)
        if not np.array_equal(forcing.reg_land.values, historical.reg_land.values):
            raise RuntimeError('Future regions differ from the historical forcing')
        if not np.array_equal(forcing.event.values, historical.event.values):
            raise RuntimeError('Future events differ from the historical forcing')
        if not np.array_equal(forcing.source_config.values, historical.source_config.values):
            raise RuntimeError('Future member order differs from the historical forcing')
    finally:
        historical.close()
    if forcing.attrs.get('n2o_mass_basis') != 'TgN_two_nitrogen_atoms_v1':
        raise RuntimeError('Future forcing is missing the N2O mass-basis label')
    years = np.asarray(forcing.year.values)
    if not np.array_equal(years, np.arange(ANCHOR_YEAR, LAST_FUTURE_YEAR + 1)):
        raise RuntimeError('Future forcing years are not 2022-2099')
    if not np.array_equal(requests.year.values, forcing.year.values) or not np.array_equal(requests.reg_land.values, forcing.reg_land.values):
        raise RuntimeError('Secondary request support differs from the future forcing')
    if tuple(str(value) for value in requests.secondary_harvest_label.values) != SECONDARY_HARVEST:
        raise RuntimeError('Secondary request labels are not secmf_harv, secyf_harv')
    for name in ('Q_sec_harvest', 'Q_sec_harvest_by_label'):
        if requests[name].attrs.get('units') != 'PgC':
            raise RuntimeError('Secondary requests must be PgC per interval')
        if not np.isfinite(requests[name]).all() or np.any(requests[name].values < 0):
            raise RuntimeError('Secondary requests are negative or nonfinite')
    if float(requests.Q_sec_harvest.isel(year=0).sum()) != 0. or float(requests.Q_sec_harvest_by_label.isel(year=0).sum()) != 0.:
        raise RuntimeError('Secondary request anchor is not zero')
    if not np.allclose(requests.Q_sec_harvest_by_label.sum('secondary_harvest_label'), requests.Q_sec_harvest, rtol=1e-12, atol=1e-14):
        raise RuntimeError('Secondary label requests do not sum to Q_sec_harvest')


def resolve_forcing(scenario, arm, copy_directory=None, atmosphere='concentration'):
    """Use the prepared forcing, or a harvest-capped copy that does not touch the source."""
    source = forcing_path(scenario)
    requests = request_path(scenario)
    forcing = xr.load_dataset(source)
    request_data = xr.load_dataset(requests)
    assert_forcing_contract(forcing, request_data)
    this_opening = secondary_opening(arm, atmosphere)
    other = 'H0' if arm == 'H1' else 'H1'
    other_opening = secondary_opening(other, atmosphere)
    difference = float(abs(this_opening - other_opening).max())
    if difference > SUPPLY_TOLERANCE_MHA:
        raise RuntimeError('H0 and H1 secondary area differ by more than the supply tolerance; '
                           'one shared land driver cannot be capped without a science choice')
    metadata = xr.load_dataset(INPUTS / 'land_metadata.nc')
    guard, capped_area = assess_secondary_supply(this_opening, forcing.event_area, metadata)
    guard.update(scenario=scenario, arm=arm, other_arm_opening_max_abs_difference_Mha=difference,
                 prepared_forcing=str(source), prepared_forcing_sha256=file_sha256(source),
                 prepared_requests=str(requests), prepared_requests_sha256=file_sha256(requests),
                 prepared_file_edited=False, working_copy=None)
    if capped_area is not None:
        if copy_directory is None:
            raise RuntimeError('Harvest cap needs an experiment directory for the working copy')
        destination = Path(copy_directory) / 'forcing' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        written = forcing.copy(deep=True)
        written['event_area'] = capped_area.transpose(*forcing.event_area.dims)
        written.event_area.attrs = dict(forcing.event_area.attrs)
        written.attrs['secondary_harvest_area_cap'] = (
            'secmf_harv and secyf_harv capped to start-of-year secdf plus incoming conversion; '
            'conversion was not edited')
        written.to_netcdf(destination)
        guard['working_copy'] = str(destination)
        forcing = written
    forcing = xr.merge([forcing, request_data[['Q_sec_harvest', 'Q_sec_harvest_by_label']]], join='exact', compat='no_conflicts')
    return forcing, guard


def supply_guard_manifest():
    """All five scenarios against both 2022 arms. Prepared files are only read."""
    metadata = xr.load_dataset(INPUTS / 'land_metadata.nc')
    openings = {arm: secondary_opening(arm) for arm in ARMS}
    difference = float(abs(openings['H0'] - openings['H1']).max())
    scenarios = {}
    capped_cells = 0
    empty_cells = 0
    for scenario in SCENARIOS:
        forcing = xr.open_dataset(forcing_path(scenario))
        try:
            assert_forcing_contract(forcing, xr.open_dataset(request_path(scenario)))
            # Identical openings make the H1 walk the shared supply. H0 is the cross-check above.
            guard, capped_area = assess_secondary_supply(openings['H1'], forcing.event_area, metadata)
        finally:
            forcing.close()
        if capped_area is not None:
            raise RuntimeError('Shared guard found a cap but was asked only to report it: ' + scenario)
        capped_cells += guard['capped_cells']
        empty_cells += guard['empty_secondary_harvest_cells']
        scenarios[scenario] = {key: guard[key] for key in (
            'cap_applied', 'capped_cells', 'empty_secondary_harvest_cells', 'minimum_slack_Mha',
            'minimum_slack_location', 'minimum_opening_minus_conversion_and_harvest_Mha', 'middle_east')}
    opening = openings['H1']
    return {
        'status': 'prepared_files_already_safe' if capped_cells == 0 and empty_cells == 0 else 'cap_required',
        'cap_applied': capped_cells != 0,
        'capped_cells': capped_cells,
        'empty_secondary_harvest_cells': empty_cells,
        'prepared_files_edited': False,
        'historical_forcing_edited': False,
        'conversion_edited': False,
        'tolerance_Mha': SUPPLY_TOLERANCE_MHA,
        'supply_definition': 'start-of-year secdf area plus same-year conversion into secdf; stand harvest does not change area',
        'h0_h1_opening_max_abs_difference_Mha': difference,
        'opening_2022_secdf_Mha': {str(region): float(value) for region, value in zip(opening.reg_land.values, opening.values)},
        'member_area_spread_Mha': {arm: float(openings[arm].attrs['member_spread_Mha']) for arm in ARMS},
        'scenarios': scenarios,
    }


def slice_forcing(forcing, member_slice, end_year):
    sliced = forcing.isel(config=member_slice).sel(year=slice(ANCHOR_YEAR, end_year))
    years = np.asarray(sliced.year.values)
    if not np.array_equal(years, np.arange(ANCHOR_YEAR, end_year + 1)):
        raise RuntimeError('Sliced future forcing is not an annual anchor-through-end window')
    anchor = sliced.event_area.isel(year=0)
    if float(anchor.sum()) != 0. or float(sliced.Q_sec_harvest.isel(year=0).sum()) != 0.:
        raise RuntimeError('Integration window does not keep a zero 2022 anchor')
    return sliced


def check_restart_anchor(state, forcing, atmosphere):
    """Concentration must match prescribed 2022 CO2. Emissions keeps solved CO2."""
    if atmosphere == 'concentration':
        if not np.array_equal(np.asarray(state.D_CO2.values), np.asarray(forcing.D_CO2.isel(year=0).values)):
            raise RuntimeError('2022 restart CO2 does not match the forcing anchor')
        return
    if 'D_CO2' not in forcing or 'd_CO2' not in forcing:
        raise RuntimeError('Emissions forcing is missing the CO2 fields that must be dropped')
    for name in ('D_CH4', 'D_N2O'):
        if not np.array_equal(np.asarray(state[name].values), np.asarray(forcing[name].isel(year=0).values)):
            raise RuntimeError('2022 restart ' + name + ' does not match the prescribed pathway anchor')


def assert_prescribed_co2(output, forcing):
    if 'D_CO2' not in output:
        raise RuntimeError('Concentration output did not retain D_CO2')
    got = output.D_CO2.transpose('year', 'config')
    expected = forcing.D_CO2.transpose('year', 'config')
    if not np.array_equal(got.year.values, expected.year.values) or got.shape != expected.shape:
        raise RuntimeError('CO2 output is not on the prescribed time axis')
    if not np.isfinite(got.values).all():
        raise RuntimeError('Nonfinite CO2 in a concentration-driven continuation')
    difference = np.asarray(got.values) - np.asarray(expected.values)
    maximum = float(abs(difference).max()) if difference.size else 0.
    if maximum != 0.:
        raise RuntimeError('CO2 departed from the prescribed concentration path by ' + format(maximum, '.3g') + ' ppm')
    return {'matches_forcing': True, 'maximum_abs_difference_ppm': maximum}


def assert_solved_atmosphere(output, forcing):
    """CO2 and temperature were integrated. CH4 and N2O stay prescribed."""
    leaked = [name for name in ('D_CO2', 'd_CO2', 'D_Tg', 'D_Td') if name in forcing]
    if leaked:
        raise RuntimeError('Solved atmosphere remained in the forcing: ' + ', '.join(leaked))
    finite = {}
    for name in ('D_CO2', 'D_Tg', 'D_Tl'):
        if name not in output or not np.isfinite(np.asarray(output[name].values)).all():
            raise RuntimeError('Solved atmosphere is missing or nonfinite: ' + name)
        finite[name] = True
    return {'solved': True, 'finite': finite}


def run_identity(payload):
    return hashlib.sha256(json.dumps(plain(payload), sort_keys=True).encode()).hexdigest()


def execute(args, mirror=True):
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    from oscar._core._model._modules.regional_forest_secondary_carbon import PROJECTION, SEC_EXCHANGES
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    from workflows.run_regional_forest_age import SECONDARY_EVENTS, SUMMARIES, diagnostics, drop_prescribed_co2

    if tuple(SECONDARY_EVENTS) != SECONDARY_HARVEST:
        raise RuntimeError('History runner secondary events differ from secmf_harv, secyf_harv')
    if args.atmosphere not in ATMOSPHERES:
        raise ValueError('Unknown atmosphere')
    oscar_file = require_study_oscar()
    output = claim_directory(args.output)
    if mirror:
        mirror_stdio(output / 'driver.log')
    member_slice = slice(args.member_start, args.member_stop)
    partial = (args.member_start, args.member_stop) != (0, 500) or args.end_year != LAST_FUTURE_YEAR
    history, state, parameters = load_restart(args.arm, member_slice, args.atmosphere)
    if args.forcing:
        forcing = xr.load_dataset(args.forcing)
        request_data = xr.load_dataset(request_path(args.scenario))
        assert_forcing_contract(forcing if forcing.sizes.get('year', 0) == 78 else xr.load_dataset(forcing_path(args.scenario)), request_data)
        # A parent working copy is already the full prepared file plus a harvest cap.
        if 'Q_sec_harvest' not in forcing:
            forcing = xr.merge([forcing, request_data[['Q_sec_harvest', 'Q_sec_harvest_by_label']]], join='exact')
        guard, capped_area = assess_secondary_supply(
            secondary_opening(args.arm, args.atmosphere), forcing.event_area, xr.load_dataset(INPUTS / 'land_metadata.nc'))
        if capped_area is not None:
            raise RuntimeError('Working forcing still exceeds secondary supply')
        guard.update(working_copy=str(Path(args.forcing).resolve()), prepared_file_edited=False, cap_applied=False)
    else:
        forcing, guard = resolve_forcing(args.scenario, args.arm, output, args.atmosphere)
    json_write(output / 'harvest_supply_guard.json', guard)
    forcing = slice_forcing(forcing, member_slice, args.end_year)
    if not np.array_equal(forcing.source_config.values, state.source_config.values):
        raise RuntimeError('Forcing member slice does not match the 2022 restart')
    check_restart_anchor(state, forcing, args.atmosphere)
    if args.atmosphere == 'emissions':
        forcing = drop_prescribed_co2(forcing)
    age_on = args.arm == 'H1'
    _, layout, kernels, _ = load_runtime_inputs(
        INPUTS, births=BIRTHS, tau_years=TAU_YEARS, age_enabled=age_on, member_slice=member_slice)
    if not np.array_equal(state.birth_time.values, np.asarray(layout.births)):
        raise RuntimeError('2022 birth axis does not match the history birth slots; refusing to add an age rule')
    layout.validate(state, float(ANCHOR_YEAR))
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = {
        'scenario': args.scenario, 'arm': args.arm, 'tau_years': TAU_YEARS,
        'age_production': 'on' if age_on else 'off', 'history_policy': HISTORY_POLICY,
        'harvest_contract': 'secondary_gross_agb_request_proportional', 'nt': NT,
        'dtype': 'float64', 'scheme': 'imex', 'adapt_nt': False,
        'member_start': args.member_start, 'member_stop': args.member_stop,
        'anchor_year': ANCHOR_YEAR, 'end_year': args.end_year, 'atmosphere': args.atmosphere,
        'restart': str((history_directory(args.arm, args.atmosphere) / 'final_state.nc').resolve()),
        'history_run_identity': history['run_identity'], 'smoke': bool(args.smoke),
    }
    identity = run_identity({'configuration': config, 'code': code, 'guard': guard['prepared_forcing_sha256'] if 'prepared_forcing_sha256' in guard else guard.get('working_copy')})
    if args.atmosphere == 'concentration':
        scope = 'Concentration-driven continuation from this arm 2022 state. 2022 is the anchor; later years are integrated.'
    else:
        scope = 'Emissions-driven continuation from this arm emissions 2022 state. CO2 and temperature are solved; later years are integrated.'
    report = {
        'status': 'running', 'run_identity': identity, 'code_commit': code, 'oscar_file': oscar_file,
        'configuration': config, 'harvest_supply_guard': guard, 'source_config': state.source_config.values.tolist(),
        'members': int(state.sizes['config']), 'phases': [],
        'scope': scope,
    }
    json_write(output / 'status.json', report)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir()
    parameters = parameters.copy()
    parameters.attrs.update(run_identity=identity, history_policy=HISTORY_POLICY,
                            harvest_contract=config['harvest_contract'], harvest_projection=PROJECTION)
    parameters.to_netcdf(output / 'parameters.nc')
    summaries = list(SUMMARIES) + list(AGE_EXCHANGES) + list(HARVEST_EXCHANGES) + list(SEC_EXCHANGES)
    retained = summaries + ['D_CO2', 'D_CH4', 'D_Tg', 'D_Pg', 'D_Tl', 'r_Pl', 'D_Ebb_CH4']
    started = time.perf_counter()
    try:
        model = assemble_native_forest_age_model(
            layout, age_production=kernels, secondary_agb_events=SECONDARY_HARVEST, harvest_projection=PROJECTION)
        before = state
        output_data, final = model(before, parameters, forcing, dtype=np.float64, var_keep=retained,
                                   keep_prog=False, get_final=True, nt=NT, adapt_nt=False)
        if int(final.year) != args.end_year:
            raise RuntimeError('Continuation did not end on the requested year')
        budget = diagnostics(before, final, output_data)
        if args.atmosphere == 'concentration':
            co2 = assert_prescribed_co2(output_data, forcing)
        else:
            co2 = assert_solved_atmosphere(output_data, forcing)
        for name, value in final.data_vars.items():
            if name.startswith('rf_') and not np.isfinite(value.values).all():
                raise RuntimeError('Nonfinite physical restart state: ' + name)
        if args.smoke:
            tolerance = float(history['startup_check']['budget_diagnostic_tolerance_PgC'])
            residual = budget['maximum_cumulative_budget_residual_PgC']
            budget['historical_startup_tolerance_PgC'] = tolerance
            budget['historical_startup_residual_PgC'] = float(history['startup_check']['maximum_cumulative_budget_residual_PgC'])
            if residual > tolerance:
                raise RuntimeError('Carbon-budget residual is worse than the historical startup check')
        record = {'start': ANCHOR_YEAR, 'end': args.end_year, 'elapsed_seconds': time.perf_counter() - started, **budget}
        report['phases'].append(record)
        report['prescribed_co2' if args.atmosphere == 'concentration' else 'solved_atmosphere'] = co2

        def save(dataset, path):
            dataset.attrs.update(run_identity=identity, history_policy=HISTORY_POLICY,
                                 harvest_contract=config['harvest_contract'], harvest_projection=PROJECTION,
                                 scenario=args.scenario, arm=args.arm)
            dataset.to_netcdf(path)

        save(output_data, output / 'annual_summary.nc')
        save(final, checkpoints / f'state_{args.end_year}.nc')
        save(final, output / 'final_state.nc')
        if args.smoke:
            status = 'passed_future_smoke'
        elif partial:
            status = 'passed_future_batch'
        else:
            status = 'passed_future'
        report.update(status=status, elapsed_seconds=time.perf_counter() - started)
        json_write(output / 'report.json', report)
        json_write(output / 'status.json', report)
        with (output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(record)) + '\n')
        print(json.dumps(plain({'status': status, 'prescribed_co2': co2, **record})), flush=True)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                      elapsed_seconds=time.perf_counter() - started)
        json_write(output / 'status.json', report)
        raise


def orchestrate(args):
    from workflows.run_regional_forest_age_batches import merge_member_datasets

    if (args.member_start, args.member_stop) != (0, 500) or args.end_year != LAST_FUTURE_YEAR:
        raise ValueError('Batch continuation runs the full 500-member window through 2099')
    if args.smoke:
        raise ValueError('Smoke is a single process, not a 500-member batch')
    output = claim_directory(args.output)
    mirror_stdio(output / 'driver.log')
    require_study_oscar()
    history = history_report(args.arm, args.atmosphere)
    forcing, guard = resolve_forcing(args.scenario, args.arm, output, args.atmosphere)
    forcing.close() if hasattr(forcing, 'close') else None
    json_write(output / 'harvest_supply_guard.json', guard)
    used = Path(guard['working_copy']) if guard.get('working_copy') else forcing_path(args.scenario)
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = {
        'scenario': args.scenario, 'arm': args.arm, 'tau_years': TAU_YEARS,
        'age_production': 'on' if args.arm == 'H1' else 'off', 'history_policy': HISTORY_POLICY,
        'harvest_contract': 'secondary_gross_agb_request_proportional', 'nt': NT, 'dtype': 'float64',
        'batch_size': args.batch_size, 'jobs': args.jobs, 'end_year': args.end_year,
        'atmosphere': args.atmosphere,
        'restart': str((history_directory(args.arm, args.atmosphere) / 'final_state.nc').resolve()),
        'history_run_identity': history['run_identity'], 'forcing': str(used),
    }
    identity = run_identity({'configuration': config, 'code': code, 'forcing_sha256': file_sha256(used)})
    batches = output / 'batches'
    launch = output / 'launch'
    batches.mkdir()
    launch.mkdir()
    chunks = [(first, min(first + args.batch_size, 500), batches / f'{first:03d}_{min(first + args.batch_size, 500):03d}')
              for first in range(0, 500, args.batch_size)]
    report = {'status': 'running', 'run_identity': identity, 'code_commit': code, 'configuration': config,
              'harvest_supply_guard': guard, 'members': 500, 'completed_batches': []}
    json_write(output / 'status.json', report)
    started = time.perf_counter()

    def spawn(chunk):
        first, last, directory = chunk
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                   '--scenario', args.scenario, '--arm', args.arm, '--atmosphere', args.atmosphere,
                   '--output', str(directory),
                   '--member-start', str(first), '--member-stop', str(last),
                   '--end-year', str(args.end_year), '--batch-size', str(args.batch_size),
                   '--forcing', str(used), '--worker']
        label = f'{first:03d}_{last:03d}'
        json_write(launch / (label + '.command.json'), command)
        environment = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
        with (launch / (label + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        json_write(launch / (label + '.exit.json'), {'exit_code': result.returncode})
        return label, result.returncode

    from concurrent.futures import ThreadPoolExecutor, as_completed
    failed = []
    try:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(chunks))) as pool:
            futures = [pool.submit(spawn, chunk) for chunk in chunks]
            for future in as_completed(futures):
                label, exit_code = future.result()
                report['completed_batches'].append({'batch': label, 'exit_code': exit_code})
                json_write(output / 'status.json', report)
                if exit_code != 0:
                    failed.append(label)
        if failed:
            raise RuntimeError('Incomplete continuation: failed batches ' + ','.join(failed))
        records = []
        for first, last, directory in chunks:
            child = json.loads((directory / 'report.json').read_text())
            if child['status'] != 'passed_future_batch' or child['code_commit'] != code:
                raise RuntimeError('A batch failed or used a different code snapshot')
            child_config = child['configuration']
            for key in ('scenario', 'arm', 'atmosphere', 'tau_years', 'age_production', 'history_policy', 'harvest_contract', 'end_year'):
                if child_config[key] != config[key] and not (key == 'tau_years' and float(child_config[key]) == float(config[key])):
                    raise RuntimeError('Batch scientific settings differ: ' + key)
            if (child_config['member_start'], child_config['member_stop']) != (first, last):
                raise RuntimeError('Batch bounds differ')
            if child_config['history_run_identity'] != history['run_identity']:
                raise RuntimeError('Batch restart identity differs')
            records.append({'directory': str(directory), 'run_identity': child['run_identity'],
                            'member_start': first, 'member_stop': last, 'elapsed_seconds': child['elapsed_seconds']})
        reference = xr.open_dataset(INPUTS / 'primary_parameters.nc')
        products = ('parameters.nc', 'annual_summary.nc', 'final_state.nc', f'checkpoints/state_{args.end_year}.nc')
        (output / 'checkpoints').mkdir(exist_ok=True)
        for relative in products:
            opened = [xr.load_dataset(directory / relative) for _, _, directory in chunks]
            try:
                for dataset, child in zip(opened, records):
                    if dataset.attrs.get('run_identity') != child['run_identity']:
                        raise RuntimeError('A batch file has a different run identity')
                    if relative == 'annual_summary.nc' and not np.array_equal(dataset.year.values, np.arange(ANCHOR_YEAR, args.end_year + 1)):
                        raise RuntimeError('Batch annual time axis is not 2022 through 2099')
                    if relative != 'parameters.nc' and relative != 'annual_summary.nc':
                        if dataset.year.ndim != 0 or int(dataset.year) != args.end_year:
                            raise RuntimeError('Batch final year is not 2099')
                merged = merge_member_datasets(opened, reference, identity, member_geometry=False)
                merged.attrs.update(history_policy=HISTORY_POLICY, scenario=args.scenario, arm=args.arm,
                                    harvest_contract=config['harvest_contract'])
                merged.to_netcdf(output / relative)
            finally:
                for dataset in opened:
                    dataset.close()
        reference.close()
        if args.atmosphere == 'concentration':
            scope = 'Five-scenario concentration continuation is one scenario-arm of that set'
        else:
            scope = 'Five-scenario emissions continuation is one scenario-arm of that set'
        report.update(status='passed_future', batches=records, elapsed_seconds=time.perf_counter() - started,
                      scope=scope)
        json_write(output / 'report.json', report)
        json_write(output / 'status.json', report)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error),
                      elapsed_seconds=time.perf_counter() - started)
        json_write(output / 'status.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--atmosphere', choices=ATMOSPHERES, default='concentration',
                        help='concentration prescribes CO2; emissions solves CO2 and temperature from the matching emissions history')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--member-start', type=int, default=0)
    parser.add_argument('--member-stop', type=int, default=500)
    parser.add_argument('--end-year', type=int, default=LAST_FUTURE_YEAR)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--forcing', type=Path, help='Already checked forcing. Skips writing another cap copy.')
    parser.add_argument('--worker', action='store_true', help='Run this member slice in-process.')
    parser.add_argument('--smoke', action='store_true', help='Require the historical startup carbon-budget bar.')
    parser.add_argument('--guard-only', action='store_true', help='Write the shared supply manifest and do not integrate.')
    args = parser.parse_args()
    if args.guard_only:
        FUTURE_ROOT.mkdir(parents=True, exist_ok=True)
        manifest = supply_guard_manifest()
        destination = FUTURE_ROOT / 'secondary_harvest_supply_guard.json'
        json_write(destination, manifest)
        print(json.dumps({'manifest': str(destination), 'status': manifest['status'],
                          'capped_cells': manifest['capped_cells'],
                          'empty_secondary_harvest_cells': manifest['empty_secondary_harvest_cells']}), flush=True)
        return 0
    if args.scenario is None or args.arm is None or args.output is None:
        raise ValueError('A continuation needs --scenario, --arm and --output')
    if not 1 <= args.batch_size <= 500 or args.jobs < 1:
        raise ValueError('Positive batch size and worker count required')
    if not 0 <= args.member_start < args.member_stop <= 500:
        raise ValueError('Member bounds must be a contiguous part of the original 500')
    if not ANCHOR_YEAR < args.end_year <= LAST_FUTURE_YEAR:
        raise ValueError('End year must be inside 2023-2099')
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    span = args.member_stop - args.member_start
    if args.worker or span <= args.batch_size:
        execute(args, mirror=not args.worker)
    else:
        orchestrate(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one v2 restoration cell from the common H1 emissions 2022 restart.

Preparation and restart auditing read inputs without model integration.
A HOLD.md marker blocks model execution until its required dependencies are available.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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
from workflows.prepare_restoration_overshoot import (  # noqa: E402
    ANCHOR, END, HISTORY, INPUTS, PACKAGE, PLAN, SCENARIOS, STARTS, SPANS,
    audit_schedule, digest, relative, require, state_areas, write_json,
)
from workflows.run_regional_forest_age_future import (  # noqa: E402
    NT, TAU_YEARS, SECONDARY_HARVEST, assert_solved_atmosphere,
    check_restart_anchor, claim_directory, history_report, mirror_stdio,
    require_study_oscar, run_identity,
)

SCHEMA = 'restoration-overshoot-plan-v2'
MODES = ('on', 'off')
BIRTHS = np.arange(850.5, float(END), 1.)
HOLD = PLAN.parent / 'HOLD.md'
RESTART_AUDIT = PLAN.parent / 'RESTART_SUPPLY_AUDIT.json'


def load_plan(path=PLAN):
    path = Path(path)
    require(path.resolve() == PLAN.resolve(), 'Use the current v2 PLAN at its fixed project path')
    plan = json.loads(path.read_text())
    require(plan.get('schema') == SCHEMA, 'Unsupported restoration PLAN schema; expected restoration-overshoot-plan-v2')
    expected = {(s, y, n, a) for s in SCENARIOS for y in STARTS for n in SPANS for a in MODES}
    actual = [(c['scenario'], c['start'], c['span'], c['age_production']) for c in plan['cells']]
    require(len(actual) == 280 and set(actual) == expected, 'PLAN is not the prescribed 280-cell matrix')
    require(plan['years'] == [ANCHOR, END] and plan['land_fraction'] == 0.125, 'Unexpected restoration design')
    for cell in plan['cells']:
        start, span = cell['start'], cell['span']
        require(cell.get('calendar_interval') == [start, start + span] and
                cell.get('native_end_years') == list(range(start + 1, start + span + 1)) and
                cell.get('birth_times') == [y - 0.5 for y in range(start + 1, start + span + 1)],
                'PLAN must map calendar starts to native end-year labels explicitly')
    ids = plan['source_config']
    require(len(ids) == 500 and len(set(ids)) == 500, 'PLAN must retain all original 500 members')
    with xr.open_dataset(INPUTS / 'primary_parameters.nc') as original:
        require(original.source_config.values.tolist() == ids, 'PLAN reordered original member IDs')
        require(np.array_equal(original.config, np.arange(500)), 'Unexpected original config coordinate')
    require(plan['input_manifest_sha256'] == digest(PACKAGE / 'manifest.json'), 'Input manifest changed')
    manifest = json.loads((PACKAGE / 'manifest.json').read_text())
    require(manifest['schema'] == 'restoration-overshoot-inputs-v2', 'Input package is not v2')
    return plan


def select_cell(plan, scenario, start, span, mode):
    cells = [c for c in plan['cells'] if (c['scenario'], c['start'], c['span'], c['age_production'])
             == (scenario, start, span, mode)]
    require(len(cells) == 1, 'Requested cell is not uniquely defined by v2 PLAN')
    return cells[0]


def land_with_schedule(cell):
    with xr.open_dataset(PACKAGE / 'background_forcing.nc') as source:
        area = source.event_area.load().copy(deep=True)
    with xr.open_dataset(PACKAGE / 'establishment_flux.nc') as schedules:
        flow = schedules.establishment_flux.sel(schedule=cell['schedule'], drop=True).load()
    area.loc[{'event': flow.event.values}] = flow.transpose('year', 'reg_land', 'event')
    return area


def compose_forcing(cell, member_slice, *, solved=True):
    """One land schedule, VL drivers, and only the named D_Eff background."""
    forcing = xr.load_dataset(PACKAGE / 'background_forcing.nc').isel(config=member_slice)
    forcing['event_area'] = land_with_schedule(cell)
    with xr.open_dataset(PACKAGE / 'fossil_backgrounds.nc') as backgrounds:
        forcing['D_Eff'] = backgrounds.D_Eff.sel(scenario=cell['scenario'], drop=True).isel(config=member_slice, missing_dims='ignore').load()
    with xr.open_dataset(PACKAGE / 'secondary_requests.nc') as requests:
        forcing = xr.merge([forcing, requests.load()], join='exact', compat='equals')
    require(np.array_equal(forcing.year, np.arange(ANCHOR, END + 1)), 'Incomplete 2022-2099 forcing')
    require(float(abs(forcing.event_area.isel(year=0)).sum()) == 0.0, 'Nonzero structural land anchor')
    require(float(abs(forcing.Q_sec_harvest.isel(year=0)).sum()) == 0.0, 'Nonzero harvest anchor')
    if solved:
        forcing = forcing.drop_vars([n for n in ('D_CO2', 'd_CO2', 'D_Tg', 'D_Td') if n in forcing])
    return forcing


def restart_areas(plan):
    """Read actual emissions class areas for every member, never a planning proxy."""
    report = history_report('H1', 'emissions')
    final, parameters, checkpoint = (HISTORY / 'final_state.nc', HISTORY / 'parameters.nc',
                                     HISTORY / 'checkpoints/state_2022.nc')
    require(all(p.is_file() for p in (final, parameters, checkpoint)), 'H1 emissions restart is not fully published')
    metadata = xr.load_dataset(INPUTS / 'land_metadata.nc')
    with xr.open_dataset(final) as state, xr.open_dataset(parameters) as p, xr.open_dataset(checkpoint) as check:
        require(state.year.ndim == 0 and float(state.year) == ANCHOR, 'Restart is not the 2022 state')
        for label, dataset in (('state', state), ('parameters', p), ('checkpoint', check)):
            require(dataset.sizes.get('config') == 500, 'Incomplete 500-member ' + label)
            require(dataset.source_config.values.tolist() == plan['source_config'], 'Reordered/missing members in ' + label)
            require(np.array_equal(dataset.config, np.arange(500)), 'Changed config order in ' + label)
        require(np.array_equal(state.D_CO2.values, check.D_CO2.values), 'Restart/checkpoint CO2 differs')
        area = state_areas(state, metadata)
    return report, area, metadata


def audit_restart(plan):
    """Audit all 28 schedules against the real 500-member H1 emissions state."""
    history, area, metadata = restart_areas(plan)
    records = []
    for cell in plan['cells']:
        if cell['scenario'] != 'VL' or cell['age_production'] != 'on':
            continue
        audit, _ = audit_schedule(land_with_schedule(cell), area, metadata)
        records.append(dict(schedule=cell['schedule'], **audit))
    require(len(records) == 28, 'Missing schedules in actual restart audit')
    record = {'status': 'passed_data_audit', 'model_run': False, 'members': 500,
              'source_config': plan['source_config'], 'history_run_identity': history['run_identity'],
              'input_manifest_sha256': plan['input_manifest_sha256'], 'windows': records,
              'scope': 'Actual H1 emissions 2022 class areas. All 28 source-supply budgets pass without caps.'}
    write_json(RESTART_AUDIT, record)
    plan['history_run_identity'] = history['run_identity']
    plan['restart_dependency'] = {'ready': True, 'history_directory': relative(HISTORY),
                                  'history_run_identity': history['run_identity'], 'members': 500,
                                  'scope': record['scope']}
    plan['restart_supply_audit'] = relative(RESTART_AUDIT)
    plan['restart_supply_audit_sha256'] = digest(RESTART_AUDIT)
    plan['status'] = 'audited_on_hold' if HOLD.exists() else 'audited_not_launched'
    write_json(PLAN, plan)
    return record


def validate_restart_identity(plan):
    require(plan['restart_dependency'].get('ready') is True, 'Run --audit-restart after H1 emissions history is published')
    history = history_report('H1', 'emissions')
    require(history['run_identity'] == plan['history_run_identity'], 'H1 emissions restart identity changed; repeat data audit')
    require(RESTART_AUDIT.is_file(), 'Actual 500-member restart audit is missing')
    require(digest(RESTART_AUDIT) == plan.get('restart_supply_audit_sha256'), 'Restart audit changed')
    audit = json.loads(RESTART_AUDIT.read_text())
    require(audit['members'] == 500 and len(audit['windows']) == 28, 'Incomplete restart supply audit')
    require(audit['history_run_identity'] == history['run_identity'] and
            audit['input_manifest_sha256'] == plan['input_manifest_sha256'] and
            audit['source_config'] == plan['source_config'], 'Restart audit is for another design or ensemble')
    return history


def require_launchable(plan):
    require(not HOLD.exists(), 'Restoration HOLD is active while required baseline dependencies are incomplete')
    return validate_restart_identity(plan)


def load_restart_slice(plan, member_slice):
    """Slice before loading large restart fields; retain original parameters."""
    history = require_launchable(plan)
    with xr.open_dataset(HISTORY / 'final_state.nc') as source:
        state = source.isel(config=member_slice).load()
    with xr.open_dataset(HISTORY / 'parameters.nc') as source:
        parameters = source.isel(config=member_slice).load()
    expected = plan['source_config'][member_slice]
    require(state.source_config.values.tolist() == expected and parameters.source_config.values.tolist() == expected,
            'Restart slice differs from the original ordered member IDs')
    require(state.year.ndim == 0 and float(state.year) == ANCHOR, 'Restart slice is not 2022')
    require('k_F_AG' in parameters and 'k_F_AG_leaf_only' in parameters, 'Missing secondary carbon projection')
    return history, state, parameters


def configuration(args, cell, history):
    return {'scenario': cell['scenario'], 'start': cell['start'], 'span': cell['span'],
            'schedule': cell['schedule'], 'age_production': cell['age_production'],
            'calendar_interval': cell['calendar_interval'], 'native_end_years': cell['native_end_years'],
            'birth_times': cell['birth_times'],
            'age_clock': None, 'tau_years': TAU_YEARS, 'history_policy': 'area_proportional',
            'harvest_contract': 'secondary_gross_agb_request_proportional', 'nt': NT,
            'dtype': 'float64', 'scheme': 'imex', 'adapt_nt': False, 'atmosphere': 'emissions',
            'anchor_year': ANCHOR, 'end_year': END, 'member_start': args.member_start,
            'member_stop': args.member_stop, 'history_run_identity': history['run_identity'],
            'restart': relative(HISTORY / 'final_state.nc'), 'input_manifest_sha256': args.plan_data['input_manifest_sha256'],
            'land_fraction': 0.125, 'land_cap': False, 'background_rule': 'D_Eff only; otherwise VL'}


def initial_report(args, cell, history):
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config = configuration(args, cell, history)
    runner = digest(Path(__file__))
    identity = run_identity({'configuration': config, 'code': code, 'runner_sha256': runner})
    return {'status': 'running', 'run_identity': identity, 'code_commit': code, 'runner_sha256': runner,
            'configuration': config, 'members': args.member_stop - args.member_start,
            'source_config': args.plan_data['source_config'][args.member_start:args.member_stop],
            'scope': 'Common H1 emissions 2022 state; production on/off from 2023. Controlled D_Eff backgrounds, common VL remaining drivers.'}


def execute(args, cell):
    plan = args.plan_data
    require_launchable(plan)  # This precedes solver imports and output-directory creation.
    from oscar._core._model._modules.regional_forest_age_diagnostics import AGE_EXCHANGES, HARVEST_EXCHANGES
    from oscar._core._model._modules.regional_forest_age_reference import assemble_native_forest_age_model
    from oscar._core._model._modules.regional_forest_secondary_carbon import PROJECTION, SEC_EXCHANGES
    from workflows.prepare_regional_forest_age_inputs import load_runtime_inputs
    from workflows.run_regional_forest_age import SUMMARIES, diagnostics, extend_birth_slots

    member_slice = slice(args.member_start, args.member_stop)
    history, state, parameters = load_restart_slice(plan, member_slice)
    forcing = compose_forcing(cell, member_slice, solved=False)
    require(forcing.source_config.values.tolist() == state.source_config.values.tolist(), 'Forcing and restart members differ')
    check_restart_anchor(state, forcing, 'emissions')
    metadata = xr.load_dataset(INPUTS / 'land_metadata.nc')
    supply, _ = audit_schedule(forcing.event_area, state_areas(state, metadata), metadata)
    forcing = forcing.drop_vars([n for n in ('D_CO2', 'd_CO2', 'D_Tg', 'D_Td') if n in forcing])
    _, layout, kernels, _ = load_runtime_inputs(INPUTS, births=BIRTHS, tau_years=TAU_YEARS,
                                               age_enabled=cell['age_production'] == 'on', member_slice=member_slice)
    require((kernels is None) == (cell['age_production'] == 'off'), 'Production switch is inconsistent')
    state = extend_birth_slots(state, layout, year=ANCHOR)
    oscar_file = require_study_oscar()
    output = claim_directory(args.output)
    if not args.worker:
        mirror_stdio(output / 'driver.log')
    report = initial_report(args, cell, history)
    report.update(oscar_file=oscar_file, source_supply_audit=supply)
    write_json(output / 'status.json', report)
    (output / 'checkpoints').mkdir()
    identity = report['run_identity']
    attrs = {'run_identity': identity, 'age_production': cell['age_production'],
             'scenario': cell['scenario'], 'schedule': cell['schedule'],
             'history_run_identity': history['run_identity'], 'history_policy': 'area_proportional',
             'harvest_contract': 'secondary_gross_agb_request_proportional', 'harvest_projection': PROJECTION}
    parameters = parameters.copy()
    parameters.attrs.update(attrs)
    parameters.to_netcdf(output / 'parameters.nc')
    summaries = list(SUMMARIES) + list(AGE_EXCHANGES) + list(HARVEST_EXCHANGES) + list(SEC_EXCHANGES)
    retained = list(dict.fromkeys(summaries + ['D_CO2', 'D_CH4', 'D_Tg', 'D_Pg', 'D_Tl', 'r_Pl', 'D_Ebb_CH4']))
    started = time.perf_counter()
    try:
        model = assemble_native_forest_age_model(layout, age_production=kernels, age_clock=None,
                                                 secondary_agb_events=SECONDARY_HARVEST, harvest_projection=PROJECTION)
        annual, final = model(state, parameters, forcing, dtype=np.float64, var_keep=retained,
                              keep_prog=False, get_final=True, nt=NT, adapt_nt=False)
        require(int(final.year) == END, 'Restoration did not reach 2099')
        budget = diagnostics(state, final, annual)
        solved = assert_solved_atmosphere(annual, forcing)
        for name, value in final.data_vars.items():
            if name.startswith('rf_'):
                require(np.isfinite(value.values).all(), 'Nonfinite physical restart state: ' + name)
        annual = annual.sel(year=slice(ANCHOR + 1, END))
        require(np.array_equal(annual.year, np.arange(ANCHOR + 1, END + 1)), 'Incomplete annual-mean output')
        # D_Eff is held constant on each native annual interval. Retain the
        # actual supplied driver alongside the already retained D_Eluc.
        annual['D_Eff'] = forcing.D_Eff.sel(year=annual.year)
        annual.D_Eff.attrs.update(time_convention='Supplied annual interval rate, constant within the native step')
        annual['D_Fland_net'] = annual.D_Fland - annual.D_Eluc
        annual.D_Fland_net.attrs.update(units='PgC yr-1',
            definition='Net land uptake anomaly entering the atmospheric budget, D_Fland minus D_Eluc; all native land classes and carbon pools, excluding permafrost')
        annual.attrs.update(attrs, time_convention='Annual model means for 2023-2099; 2022 analysis anchor is the same H1 emissions historical annual mean')
        final.attrs.update(attrs)
        annual.to_netcdf(output / 'annual_summary.nc')
        final.to_netcdf(output / 'final_state.nc')
        final.to_netcdf(output / f'checkpoints/state_{END}.nc')
        status = 'passed_restoration_batch' if (args.member_start, args.member_stop) != (0, 500) else 'passed_restoration'
        report.update(status=status, diagnostics=budget, solved_atmosphere=solved,
                      elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'report.json', report)
        write_json(output / 'status.json', report)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'status.json', report)
        raise


def orchestrate(args, cell):
    history = require_launchable(args.plan_data)
    # The actual full restart is checked at the run entrance, not the planning proxy.
    _, area, metadata = restart_areas(args.plan_data)
    supply, _ = audit_schedule(land_with_schedule(cell), area, metadata)
    from workflows.run_regional_forest_age_batches import merge_member_datasets

    output = claim_directory(args.output)
    mirror_stdio(output / 'driver.log')
    report = initial_report(args, cell, history)
    report.update(source_supply_audit=supply, batch_size=args.batch_size, jobs=args.jobs, completed_batches=[])
    write_json(output / 'status.json', report)
    batches, launch = output / 'batches', output / 'launch'
    batches.mkdir()
    launch.mkdir()
    chunks = [(i, min(i + args.batch_size, 500), batches / f'{i:03d}_{min(i + args.batch_size, 500):03d}')
              for i in range(0, 500, args.batch_size)]
    started = time.perf_counter()

    def spawn(chunk):
        first, last, directory = chunk
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--scenario', cell['scenario'],
                   '--start-year', str(cell['start']), '--span', str(cell['span']), '--age-production', cell['age_production'],
                   '--output', str(directory), '--member-start', str(first), '--member-stop', str(last), '--worker']
        label = f'{first:03d}_{last:03d}'
        write_json(launch / (label + '.command.json'), command)
        with (launch / (label + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=str(ROOT)),
                                    stdout=stream, stderr=subprocess.STDOUT)
        write_json(launch / (label + '.exit.json'), {'exit_code': result.returncode})
        return label, result.returncode

    try:
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(chunks))) as pool:
            futures = [pool.submit(spawn, chunk) for chunk in chunks]
            for future in as_completed(futures):
                label, exit_code = future.result()
                report['completed_batches'].append({'batch': label, 'exit_code': exit_code})
                write_json(output / 'status.json', report)
        require(all(c['exit_code'] == 0 for c in report['completed_batches']), 'One or more restoration batches failed')
        records = []
        for first, last, directory in chunks:
            child = json.loads((directory / 'report.json').read_text())
            expected_status = 'passed_restoration' if (first, last) == (0, 500) else 'passed_restoration_batch'
            require(child['status'] == expected_status, 'Restoration batch did not pass')
            require(child['runner_sha256'] == report['runner_sha256'] and child['code_commit'] == report['code_commit'], 'Batch used different code')
            expected = dict(report['configuration'], member_start=first, member_stop=last)
            require(child['configuration'] == expected, 'Batch configuration differs')
            records.append(child)
        (output / 'checkpoints').mkdir()
        with xr.open_dataset(INPUTS / 'primary_parameters.nc') as reference:
            for product in ('parameters.nc', 'annual_summary.nc', 'final_state.nc', f'checkpoints/state_{END}.nc'):
                opened = [xr.load_dataset(directory / product) for _, _, directory in chunks]
                try:
                    for dataset, child in zip(opened, records):
                        require(dataset.attrs.get('run_identity') == child['run_identity'], 'Batch file has a different identity')
                        if product == 'annual_summary.nc':
                            require(np.array_equal(dataset.year, np.arange(ANCHOR + 1, END + 1)), 'Batch annual years differ')
                        elif product != 'parameters.nc':
                            require(dataset.year.ndim == 0 and int(dataset.year) == END, 'Batch checkpoint year differs')
                    merged = merge_member_datasets(opened, reference, report['run_identity'], member_geometry=False)
                    require(merged.source_config.values.tolist() == args.plan_data['source_config'], 'Incomplete/reordered merged members')
                    merged.to_netcdf(output / product)
                finally:
                    for dataset in opened:
                        dataset.close()
        report.update(status='passed_restoration', batches=[{'run_identity': c['run_identity'], 'configuration': c['configuration']} for c in records],
                      elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'report.json', report)
        write_json(output / 'status.json', report)
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'status.json', report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-restart', action='store_true', help='Data only: audit actual H1 emissions 500-member restart against all 28 schedules')
    parser.add_argument('--scenario', choices=SCENARIOS)
    parser.add_argument('--start-year', type=int, choices=STARTS)
    parser.add_argument('--span', type=int, choices=SPANS)
    parser.add_argument('--age-production', choices=MODES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=50)
    parser.add_argument('--member-start', type=int, default=0)
    parser.add_argument('--member-stop', type=int, default=500)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.plan_data = load_plan()
    if args.audit_restart:
        record = audit_restart(args.plan_data)
        print(json.dumps({'status': record['status'], 'members': record['members'], 'schedules': len(record['windows']), 'model_run': False}), flush=True)
        return
    require_launchable(args.plan_data)
    require(None not in (args.scenario, args.start_year, args.span, args.age_production), 'Specify the v2 scenario, start, span and production switch')
    require(args.batch_size > 0 and args.jobs > 0, 'Batch size and jobs must be positive')
    require(0 <= args.member_start < args.member_stop <= 500, 'Invalid contiguous original member slice')
    cell = select_cell(args.plan_data, args.scenario, args.start_year, args.span, args.age_production)
    planned_output = (ROOT / cell['output']).resolve()
    if args.worker:
        require(args.output is not None, 'Worker output is required')
        args.output = (ROOT / args.output).resolve()
        expected = planned_output / 'batches' / f'{args.member_start:03d}_{args.member_stop:03d}'
        require(args.output == expected, 'Worker output must be its v2 planned batch directory')
        execute(args, cell)
    else:
        require((args.member_start, args.member_stop) == (0, 500), 'Formal restoration cells must contain all 500 members')
        require(args.output is None or (ROOT / args.output).resolve() == planned_output, 'Output must match the v2 PLAN')
        args.output = planned_output
        orchestrate(args, cell)


if __name__ == '__main__':
    main()

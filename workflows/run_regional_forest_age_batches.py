#!/usr/bin/env python3
"""Run one historical experiment as independent original-member batches.

Only execution is partitioned. A completed product must contain every original
member exactly once, in its original order, with either shared area geometry or explicitly member-dependent geometry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.prepare_regional_forest_age_inputs import fingerprint
from workflows.run_regional_forest_age import HISTORY_POLICIES, history_policy, validate_spinup_source, HARVEST_CONTRACTS, CANDIDATE, secondary_extension_identity

PRODUCTS = ('parameters.nc', 'prehistory_summary.nc', 'annual_summary.nc', 'final_state.nc',
            'checkpoints/state_1750.nc', 'checkpoints/state_2010.nc',
            'checkpoints/state_2020.nc', 'checkpoints/state_2022.nc')


def merge_member_datasets(datasets, expected, run_id, *, member_geometry=False):
    """Strict original-order concatenation; never average shared area states."""
    cursor = 0
    reference = datasets[0]
    for dataset in datasets:
        size = dataset.sizes['config']
        wanted = expected.isel(config=slice(cursor, cursor + size))
        for name in ('config', 'source_config'):
            if not np.array_equal(dataset[name], wanted[name]):
                raise ValueError('Missing, duplicated or reordered original members: ' + name)
        if set(dataset.data_vars) != set(reference.data_vars):
            raise ValueError('Batch output variables differ')
        for name, value in reference.data_vars.items():
            if value.attrs.get('units') != dataset[name].attrs.get('units'):
                raise ValueError('Batch output units differ: ' + name)
            if 'config' not in value.dims and not member_geometry:
                # Float summation tails are allowed, but never different area
                # histories. The first shared value is retained without averaging.
                xr.testing.assert_allclose(value, dataset[name], rtol=1e-12, atol=1e-10)
        cursor += size
    if cursor != expected.sizes['config']:
        raise ValueError('Incomplete original ensemble')
    # replace equal shared fields so harmless roundoff cannot make xarray's
    # structural compatibility test ambiguous after the scientific comparison.
    shared = [name for name in reference.data_vars if 'config' not in reference[name].dims]
    normalized = []
    for dataset in datasets:
        # One update avoids copying and aligning the full dataset per field.
        replacements = {
            name: (dataset[name].expand_dims(config=dataset.config)
                   if member_geometry else reference[name])
            for name in shared
        }
        normalized.append(dataset.assign(replacements))
    result = xr.concat(normalized, dim='config', data_vars='minimal', coords='minimal',
                       compat='equals', join='exact', combine_attrs='override')
    result.attrs['run_identity'] = run_id
    return result


def check_product_time(dataset, relative):
    """Check the declared physical period, including common mistakes in all batches."""
    years = {'prehistory_summary.nc': np.arange(850, 1751),
             'annual_summary.nc': np.arange(1750, 2023)}
    endpoints = {'final_state.nc': 2022,
                 **{f'checkpoints/state_{year}.nc': year for year in (1750, 2010, 2020, 2022)}}
    if relative in years:
        if 'year' not in dataset.coords or not np.array_equal(dataset.year.values, years[relative]):
            raise ValueError('Incomplete or shifted physical time axis: ' + relative)
    if relative in endpoints:
        if ('year' not in dataset.coords or dataset.year.ndim != 0
                or float(dataset.year) != endpoints[relative]):
            raise ValueError('Incorrect physical checkpoint year: ' + relative)


def wait_for_startup(future, directory):
    """Release other independent batches after a checked first ten-year interval."""
    while True:
        if future.done():
            _, exit_code = future.result()
            if exit_code != 0:
                raise RuntimeError('First batch failed before the startup gate')
        path = directory / 'progress.jsonl'
        if path.exists():
            lines = path.read_text().splitlines()
            if lines:
                record = json.loads(lines[0])
                if (record['start'], record['end']) != (850, 860):
                    raise ValueError('Unexpected first checked interval')
                return record
        if future.done():
            raise RuntimeError('First batch finished without the checked startup interval')
        time.sleep(1.)


def merge_products(output, chunks, primary, report, *, member_geometry=False):
    """Publish complete products only after all child reports are accepted."""
    records = []
    for first, last, directory in chunks:
        child = json.loads((directory / 'report.json').read_text())
        expected_status = 'passed_historical' if (first, last) == (0, 500) else 'passed_historical_batch'
        if child['status'] != expected_status or child['code_commit'] != report['code_commit']:
            raise ValueError('A batch failed or used a different code snapshot')
        if child['input_manifest']['sha256'] != report['input_manifest']['sha256']:
            raise ValueError('Batch input identity differs')
        if child['spinup_identity'] != report['spinup_identity']:
            raise ValueError('Batch spin-up identity differs')
        if child['source_config'] != primary.source_config.isel(config=slice(first,last)).values.tolist():
            raise ValueError('Batch source-member identity differs')
        config = child['configuration']
        if (config['member_start'], config['member_stop']) != (first, last):
            raise ValueError('Batch bounds differ')
        if history_policy(config) != history_policy(report['configuration']):
            raise ValueError('Batch historical selection policy differs')
        if config.get('harvest_contract', 'area') != report['configuration'].get('harvest_contract', 'area') or child.get('secondary_extension') != report.get('secondary_extension'):
            raise ValueError('Batch harvest contract or extension identity differs')
        if any(config[key] != report['configuration'][key] for key in ('tau_years', 'age_production', 'nt')):
            raise ValueError('Batch scientific settings differ')
        records.append({'directory': str(directory), 'run_identity': child['run_identity'],
                        'member_start': first, 'member_stop': last, 'elapsed_seconds': child['elapsed_seconds']})
    (output / 'checkpoints').mkdir(exist_ok=True)
    for relative in PRODUCTS:
        # Read and close each file once. Lazy variable-wise concatenation can
        # otherwise reopen every file for every field when batches exceed the
        # xarray file cache. The concatenation already retains these arrays.
        opened = [xr.load_dataset(directory / relative) for _, _, directory in chunks]
        try:
            for dataset, child in zip(opened, records):
                check_product_time(dataset, relative)
                if dataset.attrs.get('run_identity') != child['run_identity']:
                    raise ValueError('A batch file has a different run identity')
            merged = merge_member_datasets(opened, primary, report['run_identity'],
                                           member_geometry=member_geometry)
            merged.attrs['history_policy'] = history_policy(report['configuration'])
            merged.to_netcdf(output / relative)
        finally:
            for dataset in opened:
                dataset.close()
    report['batches'] = records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--spinup-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tau-years', type=float, required=True)
    parser.add_argument('--age-production', choices=['on', 'off'], required=True)
    parser.add_argument('--history-policy', choices=HISTORY_POLICIES, default='area_proportional')
    parser.add_argument('--harvest-contract', choices=HARVEST_CONTRACTS, default='area')
    parser.add_argument('--secondary-carbon-inputs', type=Path)
    parser.add_argument('--nt', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--jobs', type=int, default=10)
    parser.add_argument('--startup-probe', action='store_true', help='Check the first ten-year interval of one batch before releasing the remaining batches')
    parser.add_argument('--emissions-driven', action='store_true',
                        help='Integrate CO2 and temperature; do not prescribe D_CO2, d_CO2, D_Tg, or D_Td')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace an experiment directory')
    if not 1 <= args.batch_size <= 500 or args.jobs < 1:
        raise ValueError('Positive batch size and worker count required')
    secondary = args.harvest_contract == CANDIDATE
    if secondary and args.history_policy != 'area_proportional':
        raise ValueError('Secondary carbon candidate requires proportional selection')
    if secondary != (args.secondary_carbon_inputs is not None):
        raise ValueError('Secondary contract and extension directory must be specified together')
    primary = xr.load_dataset(args.inputs / 'primary_parameters.nc')
    if primary.sizes['config'] != 500 or not np.array_equal(primary.config, np.arange(500)):
        raise ValueError('Batch execution requires the complete prepared original ensemble')
    spinup = json.loads((args.spinup_directory / 'report.json').read_text())
    inputs = fingerprint(args.inputs / 'manifest.json')
    validate_spinup_source(spinup, inputs)
    extension = secondary_extension_identity(args.secondary_carbon_inputs, inputs) if secondary else None
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    identity = hashlib.sha256(json.dumps({'config':config,'code':code,'inputs':inputs['sha256'],
                                        'spinup':spinup['run_identity'], **({'secondary_extension':extension} if secondary else {})},sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True)
    batches = args.output / 'batches'; batches.mkdir()
    launch = args.output / 'launch'; launch.mkdir()
    driver_log = open(args.output / 'driver.log', 'w', buffering=1)
    class _Tee:
        def __init__(self, left, right):
            self.left, self.right = left, right
        def write(self, data):
            self.left.write(data)
            self.right.write(data)
            self.left.flush()
            self.right.flush()
        def flush(self):
            self.left.flush()
            self.right.flush()
    sys.stdout = _Tee(sys.stdout, driver_log)
    sys.stderr = _Tee(sys.stderr, driver_log)
    print('atmosphere', 'emissions' if args.emissions_driven else 'concentration',
          'jobs', args.jobs, 'batch_size', args.batch_size, flush=True)
    chunks = [(first,min(first+args.batch_size,500),batches / f'{first:03d}_{min(first+args.batch_size,500):03d}')
              for first in range(0,500,args.batch_size)]
    report = {'status':'running','run_identity':identity,'code_commit':code,'configuration':config,
              'input_manifest':inputs,'secondary_extension':extension,'spinup_identity':spinup['run_identity'],'members':500,
              'source_config':primary.source_config.values.tolist(),'completed_batches':[]}
    def save():
        (args.output / 'status.json').write_text(json.dumps(report,indent=2)+'\n')
    save(); start = time.perf_counter()
    def execute(chunk):
        first,last,directory = chunk
        command = [sys.executable,'-u','-B',str(ROOT/'workflows/run_regional_forest_age.py'),
                   '--phase','history','--inputs',str(args.inputs),'--spinup-directory',str(args.spinup_directory),
                   '--output',str(directory),'--tau-years',str(args.tau_years),'--age-production',args.age_production,
                   '--history-policy',args.history_policy,'--nt',str(args.nt),'--member-start',str(first),'--member-stop',str(last)]
        command += ['--harvest-contract', args.harvest_contract]
        if args.emissions_driven:
            command += ['--emissions-driven']
        if secondary:
            command += ['--secondary-carbon-inputs', str(args.secondary_carbon_inputs)]
        label = f'{first:03d}_{last:03d}'
        (launch / (label+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
        environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', BLIS_NUM_THREADS='1',
                           PYTHONPATH=str(ROOT))
        with (launch / (label+'.log')).open('w') as stream:
            result = subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        (launch / (label+'.exit.json')).write_text(json.dumps({'exit_code':result.returncode})+'\n')
        return label,result.returncode
    try:
        failed = []
        with ThreadPoolExecutor(max_workers=min(args.jobs,len(chunks))) as pool:
            if args.startup_probe:
                first = pool.submit(execute, chunks[0])
                report['startup_check'] = wait_for_startup(first, chunks[0][2]); save()
                futures = [first] + [pool.submit(execute,chunk) for chunk in chunks[1:]]
            else:
                futures = [pool.submit(execute,chunk) for chunk in chunks]
            for future in as_completed(futures):
                label,code = future.result()
                report['completed_batches'].append({'batch':label,'exit_code':code}); save()
                if code != 0: failed.append(label)
        if failed:
            raise RuntimeError('Incomplete ensemble: failed batches '+','.join(failed))
        merge_products(args.output,chunks,primary,report)
        report.update(status='passed_historical',elapsed_seconds=time.perf_counter()-start,
                      scope='Complete original ensemble, continuous native history; ecological and observational acceptance is separate')
        save()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    except Exception as error:
        report.update(status='failed',error=type(error).__name__+': '+str(error),elapsed_seconds=time.perf_counter()-start)
        save(); raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bind an existing corrected CMIP7 driver bundle to a canonical history.

Forcings and gross harvest requests are copied unchanged. Only the historical
source identity and its prepared continuation parameters are rebound, using
the established input conversion. Historical simulations remain immutable.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.cmip7_n2o_mass_basis import (
    BASIS, SCHEMA, correct_legacy_parameters, fingerprint, pi_identity)


def prepare(source, history_directory, output):
    source, history_directory, output = map(Path.resolve, (source, history_directory, output))
    allowed = {(ROOT / 'outputs/03_historical' / arm).resolve() for arm in ('H0', 'H1')}
    if history_directory not in allowed:
        raise ValueError('Select a canonical H0 or H1 historical publication')
    if output.exists():
        raise FileExistsError('Refusing to replace a prepared continuation binding')
    original = json.loads((source / 'manifest.json').read_text())
    history = json.loads((history_directory / 'report.json').read_text())
    if original['schema'] != SCHEMA or history['status'] != 'passed_historical' or history['members'] != 500:
        raise ValueError('Requires prepared corrected forcing and complete historical ensemble')
    if original['source_config'] != history['source_config']:
        raise ValueError('Original member identities differ')
    indexed = dict(line.split(maxsplit=1)[::-1] for line in
                   (ROOT / 'outputs/files.sha256').read_text().splitlines() if line.strip())
    report_record = fingerprint(history_directory / 'report.json')
    relative = (history_directory / 'report.json').relative_to(ROOT / 'outputs').as_posix()
    if indexed.get(relative) != report_record['sha256']:
        raise ValueError('Historical parent report differs from its accepted index')
    parameters_path = history_directory / 'parameters.nc'
    parameters = xr.load_dataset(parameters_path)
    if parameters.attrs['run_identity'] != history['run_identity']:
        raise ValueError('Historical parameter identity differs')
    np.testing.assert_array_equal(parameters.config, np.arange(500))
    np.testing.assert_array_equal(parameters.source_config, history['source_config'])
    revised = correct_legacy_parameters(parameters)
    if pi_identity(revised) != original['n2o_continuation']['pi_sha256']:
        raise ValueError('Historical reference differs from the existing forcing convention')
    state_path = history_directory / 'final_state.nc'
    state_key = state_path.relative_to(ROOT / 'outputs').as_posix()
    state_record = dict(path=str(state_path), size_bytes=state_path.stat().st_size,
                        sha256=indexed[state_key])
    parameter_record = fingerprint(parameters_path)
    revised.attrs.pop('run_identity', None)
    revised.attrs.update(source_run_identity=history['run_identity'],
        role='Prepared continuation parameters; not a new historical execution',
        source_parameters_sha256=parameter_record['sha256'],
        source_state_sha256=state_record['sha256'])
    output.mkdir(parents=True)
    files = {}
    for name in original['outputs']:
        if name.startswith('future_') or name == 'projection_parameters.nc':
            shutil.copy2(source / name, output / name)
            files[name] = fingerprint(output / name)
            if files[name]['sha256'] != original['outputs'][name]['sha256']:
                raise ValueError('Copied forcing or harvest input differs')
    revised.to_netcdf(output / 'continuation_parameters.nc',
        encoding={n: {'zlib': True, 'complevel': 2} for n in revised.data_vars})
    files['continuation_parameters.nc'] = fingerprint(output / 'continuation_parameters.nc')
    result = deepcopy(original)
    result.update(prepared_utc=datetime.now(timezone.utc).isoformat(),
        history_run_identity=history['run_identity'], outputs=files,
        inputs=dict(source_revision_manifest=fingerprint(source / 'manifest.json'),
                    historical_parameters=parameter_record),
        scope='Unchanged corrected five-scenario forcing rebound to the selected canonical historical state',
        preparation_code={Path(__file__).name: fingerprint(__file__)})
    result['n2o_continuation'] = dict(basis=BASIS, parameters=files['continuation_parameters.nc'],
        pi_sha256=pi_identity(revised), history_run_identity=history['run_identity'],
        source_parameters=parameter_record, source_state=state_record,
        source_state_digest_basis='Accepted publication index; checkpoint not rehashed',
        historical_report=report_record)
    for scenario in result['scenarios']:
        result['scenarios'][scenario]['parent_forcing'] = files['future_' + scenario + '_forcing.nc']
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='prepared_inputs_verified', history=history['run_identity'],
                         output=str(output), source=str(source)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/processed/cmip7_n2o_mass_corrected_v1')
    parser.add_argument('--history-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.history_directory, args.output)

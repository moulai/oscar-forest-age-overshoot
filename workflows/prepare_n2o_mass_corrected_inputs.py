#!/usr/bin/env python3
"""Publish one consistent N2O input revision without rewriting prior executions."""
import argparse
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
    BASIS, SCHEMA, DYNAMIC, STATIC, correct_legacy_fields,
    correct_legacy_parameters, fingerprint, label_forcing, pi_identity)

PARENT = ROOT / 'data/processed/regional_forest_age_inputs_v1'
REFERENCE = ROOT / 'data/processed/reference_background/v1'
BACKGROUND = ROOT / 'data/processed/future_background/v3'
EXTENSION = ROOT / 'data/processed/future_background/extension_v1'
HISTORY = ROOT / 'outputs/03_historical/H1'
REQUESTS = ROOT / 'data/processed/regional_secondary_harvest_carbon_future_v1'
DEFAULT = ROOT / 'data/processed/cmip7_n2o_mass_corrected_v1'


def read_json(path):
    return json.loads(Path(path).read_text())


def unchanged_except(before, after, allowed):
    if set(before) != set(after):
        raise ValueError('The correction changed the variable schema')
    for name in set(before) - set(allowed):
        xr.testing.assert_equal(before[name], after[name])


def source_state_record():
    """Reuse the accepted output index instead of hashing the large checkpoint."""
    name = '03_historical/H1/final_state.nc'
    entries = dict(line.split(maxsplit=1)[::-1] for line in
                   (ROOT / 'outputs/files.sha256').read_text().splitlines() if line.strip())
    path = ROOT / 'outputs' / name
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=entries[name])


def gridded_n2o(scenario):
    """Read the three already validated small regional mass reductions."""
    from workflows.reduce_cmip7_emissions_extension import validate_reduction
    source = read_json(ROOT / 'manifests/cmip7_background_extension_sources.json')
    rows = source['files'] if 'files' in source else source['sources']
    wanted = ('N2O_em_anthro', 'N2O_em_AIR_anthro', 'N2O_em_openburning')
    selected = [r for r in rows if r['scenario'].lower() == scenario and r['variable'] in wanted]
    if {r['variable'] for r in selected} != set(wanted):
        raise ValueError('Incomplete same-scenario N2O source identity')
    values = {r['variable']: validate_reduction(r).emission for r in selected}
    factor = (28. / 44.) / 1e9
    return {'D_Eant_N2O': (values[wanted[0]] + values[wanted[1]]) * factor,
            'D_Ebb2_N2O': values[wanted[2]] * factor}


def prepare(output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite an input revision')
    output.mkdir(parents=True)
    pm = read_json(PARENT / 'manifest.json')
    hm = read_json(HISTORY / 'report.json')
    qm = read_json(REQUESTS / 'manifest.json')
    inputs, outputs = {}, {}

    def source(label, path):
        inputs[label] = fingerprint(path)
        return xr.load_dataset(path)

    def save(name, data):
        data.to_netcdf(output / name, encoding={n: {'zlib': True, 'complevel': 2} for n in data.data_vars})
        outputs[name] = fingerprint(output / name)

    def copy(name, path):
        inputs[name + '_source'] = fingerprint(path)
        shutil.copy2(path, output / name)
        outputs[name] = fingerprint(output / name)
        if outputs[name]['sha256'] != inputs[name + '_source']['sha256']:
            raise ValueError('Unchanged input copy differs')

    primary_old = source('regional_primary', PARENT / 'primary_parameters.nc')
    primary = correct_legacy_parameters(primary_old)
    unchanged_except(primary_old, primary, STATIC)
    save('primary_parameters.nc', primary)
    copy('indexed_original_parameters.nc', REFERENCE / 'indexed_primary_parameters.nc')
    history_old = source('historical_atmosphere', REFERENCE / 'historical_concentration_forcing.nc')
    history = correct_legacy_fields(history_old)
    save('historical_concentration_forcing.nc', history)
    regional_old = source('historical_regional_forcing', PARENT / 'historical_forcing.nc')
    regional = label_forcing(correct_legacy_fields(regional_old, DYNAMIC), primary)
    unchanged_except(regional_old, regional, DYNAMIC)
    save('historical_forcing.nc', regional)
    effective_old = source('effective_reference_parameters', REFERENCE / 'effective_parameters.nc')
    effective = correct_legacy_parameters(effective_old)
    unchanged_except(effective_old, effective, STATIC + ('Ebb_N2O_pi',))
    save('effective_parameters.nc', effective)
    bridge_path = ROOT / 'data/raw/future/cmip7_preflight_v1/oscar_v4_beta2_1/forcing_hist.nc'
    bridge = correct_legacy_fields(source('official_historical_bridge', bridge_path))

    original = source('historical_parameters', HISTORY / 'parameters.nc')
    continuation = correct_legacy_parameters(original)
    unchanged_except(original, continuation, STATIC)
    for name in ('config', 'source_config'):
        np.testing.assert_array_equal(continuation[name], primary[name])
    if pi_identity(continuation) != pi_identity(primary):
        raise ValueError('Prepared and continuation PI references differ')
    state_record = source_state_record()
    with xr.open_dataset(state_record['path']) as state:
        if state.attrs['run_identity'] != hm['run_identity'] or original.attrs['run_identity'] != hm['run_identity']:
            raise ValueError('Historical endpoint identity differs')
        for name in ('config', 'source_config'):
            np.testing.assert_array_equal(state[name], primary[name])
    continuation.attrs.pop('run_identity', None)
    continuation.attrs.update(role='Prepared continuation parameters with corrected source N2O PI; not a new model execution',
        source_run_identity=hm['run_identity'], source_parameters_sha256=inputs['historical_parameters']['sha256'],
        source_state_sha256=state_record['sha256'])
    save('continuation_parameters.nc', continuation)

    legacy = source('legacy_future_atmosphere', BACKGROUND / 'atmospheric_forcing.nc')
    corrected_legacy = label_forcing(correct_legacy_fields(legacy, DYNAMIC), primary)
    save('legacy_scenarios_atmosphere.nc', corrected_legacy)
    lineage, scenarios, diagnostics = {}, {}, {}
    for scenario in ('vl', 'm', 'h', 'hl', 'ln'):
        if scenario in ('vl', 'm', 'h'):
            atmosphere = corrected_legacy.sel(scen=['scen7-' + scenario.upper()])
            original_atmosphere = legacy.sel(scen=['scen7-' + scenario.upper()])
            forcing_directory, source_manifest, request_directory = PARENT, pm, REQUESTS
            source_scenario = qm['scenarios'][scenario]
        else:
            original_atmosphere = source(scenario + '_atmosphere', EXTENSION / scenario / 'atmospheric_forcing.nc')
            atmosphere = original_atmosphere.copy(deep=True)
            absolute = gridded_n2o(scenario)
            for dynamic, static in zip(DYNAMIC, STATIC):
                years = atmosphere.year.values[atmosphere.year.values >= 2024]
                values = absolute[dynamic].interp(year=years)
                # The gridded branch was already correct in absolute mass.
                before = original_atmosphere[dynamic].sel(year=years).isel(scen=0, drop=True)
                xr.testing.assert_allclose(before.reset_coords(drop=True) + primary_old[static].reset_coords(drop=True),
                                           values.reset_coords(drop=True), rtol=1e-13, atol=1e-13)
                atmosphere[dynamic].loc[{'year': years}] = values - primary[static].reset_coords(drop=True)
                atmosphere[dynamic].loc[{'year': 2022}] = history[dynamic].sel(year=2022)
                atmosphere[dynamic].loc[{'year': 2023}] = bridge[dynamic].sel(year=2023)
            label_forcing(atmosphere, primary)
            forcing_directory = ROOT / 'data/processed/forest_timing_future_inputs' / (scenario + '_v1')
            source_manifest = read_json(forcing_directory / 'manifest.json')
            request_directory = forcing_directory
            source_scenario = source_manifest['scenarios'][scenario]
        unchanged_except(original_atmosphere, atmosphere, DYNAMIC)
        save('atmosphere_' + scenario + '.nc', atmosphere)
        forcing_name = 'future_' + scenario + '_forcing.nc'
        old = source(scenario + '_regional_forcing', forcing_directory / forcing_name)
        forcing = old.copy(deep=True)
        for name in DYNAMIC:
            forcing[name] = atmosphere[name].sel(scen='scen7-' + scenario.upper(), year=forcing.year, drop=True)
            xr.testing.assert_equal(forcing[name].sel(year=2022, drop=True), regional[name].sel(year=2022, drop=True))
        label_forcing(forcing, primary)
        unchanged_except(old, forcing, DYNAMIC)
        save(forcing_name, forcing)
        copy('future_' + scenario + '_requests.nc', request_directory / ('future_' + scenario + '_requests.nc'))
        inputs[scenario + '_source_manifest'] = fingerprint(forcing_directory / 'manifest.json')
        lineage[scenario] = source_manifest['data_lineage'][scenario]
        scenarios[scenario] = dict(source_scenario, parent_forcing=outputs[forcing_name],
            parent_forcing_digest_basis='This corrected input revision; unchanged land and harvest provenance')
        diagnostics[scenario] = {name: {str(y): float(forcing[name].sel(year=y).sum()) for y in (2022, 2023, 2024)} for name in DYNAMIC}
    copy('projection_parameters.nc', REQUESTS / 'projection_parameters.nc')
    binding = dict(basis=BASIS, parameters=outputs['continuation_parameters.nc'],
        pi_sha256=pi_identity(primary), history_run_identity=hm['run_identity'],
        source_parameters=inputs['historical_parameters'], source_state=state_record,
        source_state_digest_basis='Accepted outputs/files.sha256; large checkpoint not rehashed',
        historical_report=fingerprint(HISTORY / 'report.json'))
    manifest = dict(schema=SCHEMA, status='prepared_inputs_verified', prepared_utc=datetime.now(timezone.utc).isoformat(),
        parent_input_manifest_sha256=fingerprint(PARENT / 'manifest.json')['sha256'],
        history_extension_manifest_sha256=qm['history_extension_manifest_sha256'],
        projection_source_sha256=qm['projection_source_sha256'], history_run_identity=hm['run_identity'],
        source_config=pm['source_config'], ordered_source_ids_sha256=pm['ordered_source_ids_sha256'],
        data_lineage=lineage, scenarios=scenarios, inputs=inputs, outputs=outputs, n2o_continuation=binding,
        n2o_mass_basis=BASIS, model_integration_executed=False,
        scope='One N2O source mass-basis revision, shared PI and historical bridge; original raw and executions retained',
        consumers='The future reader must load the bound continuation parameters; legacy readers reject this schema',
        correction=dict(legacy_fields=list(DYNAMIC + STATIC), legacy_factor=2., gridded_N2O_to_N=28./44.,
            gridded_rule='Keep correct absolute mass, subtract corrected common PI; use corrected historical 2022/2023',
            unaffected='Concentrations, climate, carbon processes, land/Q, original joint ensemble and historical physical states'),
        checks=dict(unchanged_other_fields=True, common_2022_anchor=True, common_2023_bridge=True,
            original_member_order_preserved=True, static_PI_consistent=True, source_raw_unchanged=True),
        diagnostics=diagnostics, preparation_code={p.name: fingerprint(p) for p in
            (Path(__file__), ROOT / 'workflows/cmip7_n2o_mass_basis.py')})
    temporary = output / 'manifest.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(output / 'manifest.json')
    print(json.dumps(dict(status=manifest['status'], directory=str(output), outputs=len(outputs)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    prepare(parser.parse_args().output.resolve())

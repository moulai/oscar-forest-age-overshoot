#!/usr/bin/env python3
"""Prepare CMIP7 future secondary gross-AG requests from accepted regional caches.

Data preparation only: no simulation, raw raster access or parent bundle edits.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import xarray as xr

from workflows.prepare_secondary_harvest_carbon import (
    LABELS, SOURCES, PROJECTION, TIME, coordinates_equal, digest, phase_requests,
)

SOURCE_IDS = {'vl': 'UofMD-landState-vl-3-1-1', 'm': 'UofMD-landState-m-3-1',
              'h': 'UofMD-landState-h-3-1-1'}
SOURCE_VARIABLES = [
    'interval_start', 'interval_end', 'harvest_area', 'harvest_area_available',
    'harvest_carbon_raw', 'harvest_carbon_raw_available',
    'harvest_carbon_negative_absolute', 'raw_bioh_negative_count',
    'raw_bioh_min_kgC', 'harvest_source_declared_available',
    'harvest_carbon_source_declared_available',
]
OFFICIAL_URL = ('https://github.com/PCMDI/input4MIPs_CVs/blob/'
                '23f330397904de7233cd230ba256b05b5ad739c1/docs/dataset-overviews/land-use.md')


def prepare_source(source, forcing):
    """Apply the documented historical cell-negative policy to cached statistics."""
    if not np.array_equal(forcing.year, np.arange(2022, 2100)):
        raise ValueError('Expected native 2022 anchor and 2023-2099 event labels')
    coordinates_equal(source, forcing, ['reg_land'])
    if not np.array_equal(source.harvest_source, SOURCES):
        raise ValueError('Secondary source labels/order differ')
    if not np.array_equal(source.interval, np.arange(2022, 2099)):
        raise ValueError('Expected source intervals 2022-2098, each exactly once')
    if (not np.array_equal(source.interval_start, source.interval)
            or not np.array_equal(source.interval_end, source.interval + 1)):
        raise ValueError('Source interval endpoints differ')
    missing = {n: int((~np.isfinite(source[n])).sum()) for n in SOURCE_VARIABLES}
    if any(missing.values()):
        raise ValueError('Missing processed source values: ' + str(missing))
    for name in ('harvest_source_declared_available', 'harvest_carbon_source_declared_available'):
        if not bool((source[name] == 1).all()):
            raise ValueError('Source-declared unavailable secondary harvest data')
    for name in ('harvest_area', 'harvest_area_available'):
        if source[name].attrs.get('units') != 'Mha':
            raise ValueError('Unexpected source area units: ' + name)
    for name in ('harvest_carbon_raw', 'harvest_carbon_raw_available', 'harvest_carbon_negative_absolute'):
        if source[name].attrs.get('units') != 'Pg C':
            raise ValueError('Unexpected source carbon units: ' + name)
    for actual, available in [('harvest_area', 'harvest_area_available'),
                              ('harvest_carbon_raw', 'harvest_carbon_raw_available')]:
        if not np.array_equal(source[actual], source[available]):
            raise ValueError('Stored/available secondary source views differ')
    if bool((source.harvest_area < 0).any()) or bool((source.harvest_carbon_negative_absolute < 0).any()):
        raise ValueError('Negative area or negative absolute correction')
    out = source.copy(deep=True)
    correction = out.harvest_carbon_negative_absolute
    out['harvest_carbon_negative_correction'] = correction.copy()
    out.harvest_carbon_negative_correction.attrs = {
        'units': 'Pg C', 'policy': 'Historical cell-negative-to-zero rule applied to the existing regional negative-absolute statistic',
        'formula': 'regional sum of max(-raw_cell_bioh, 0) / 1e12',
        'status': 'Newly applied in this extension; the existing future cache remains signed and unchanged',
    }
    out['harvest_carbon'] = out.harvest_carbon_raw + correction
    out.harvest_carbon.attrs = {'units': 'Pg C', 'formula': 'harvest_carbon_raw + harvest_carbon_negative_correction',
                              'conversion': 'Already regional Pg C; no area factor or other carbon rescaling'}
    if bool((out.harvest_carbon < 0).any()):
        raise ValueError('Negative corrected carbon; source review required')
    out['regional_carbon_without_positive_area'] = out.harvest_carbon.where(out.harvest_area == 0, 0)
    out.regional_carbon_without_positive_area.attrs = {
        'units': 'Pg C', 'support': 'Region x source label x annual interval; not a cell-level test',
        'role': 'Provenance diagnostic only; upstream area is not the candidate area prescription',
    }
    if bool((out.regional_carbon_without_positive_area > 0).any()):
        raise ValueError('Positive regional carbon with zero regional source area; review before preparation')
    out.attrs.update(extension_role='Source audit, not model forcing',
                     carbon_policy='Signed original plus explicitly applied historical cell-negative policy; no clipping of regional totals')
    requests = phase_requests(out, forcing)
    if not bool(np.isfinite(requests.to_array()).all()):
        raise ValueError('Missing future requests')
    # Actual native For exact merge, including any auxiliary coordinates.
    xr.merge([forcing, requests], join='exact', compat='equals')
    area = forcing.event_area.sel(event=LABELS).isel(year=slice(1, None))
    area_difference = area.transpose('year', 'reg_land', 'event').values - out.harvest_area.values
    closure = out.harvest_carbon - out.harvest_carbon_raw - correction
    diagnostic = {
        'missing_count': missing,
        'request_sum_PgC': float(requests.Q_sec_harvest.sum()),
        'raw_signed_sum_PgC': float(out.harvest_carbon_raw.sum()),
        'correction_sum_PgC': float(correction.sum()),
        'correction_sum_by_label_PgC': {s: float(correction.sel(harvest_source=s).sum()) for s in SOURCES},
        'correction_closure_max_abs_PgC': float(abs(closure).max()),
        'raw_negative_cell_occurrences': int(out.raw_bioh_negative_count.sum()),
        'raw_minimum_kgC': float(out.raw_bioh_min_kgC.min()),
        'regional_carbon_without_positive_area_PgC': float(out.regional_carbon_without_positive_area.sum()),
        'cell_carbon_without_positive_area': 'not_assessed: no retained same-cell joint diagnostic; regional test cannot exclude cell anomalies',
        'parent_event_area_minus_source_max_abs_Mha': float(np.max(np.abs(area_difference))),
        'anchor_sum_PgC': float(requests.Q_sec_harvest.isel(year=0).sum()),
        'first_event_year': 2023, 'last_event_year': 2099,
        'first_source_interval': 2022, 'last_source_interval': 2098,
        'native_forcing_exact_merge': True,
    }
    return out, requests, diagnostic


def snapshot(directory):
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in directory.iterdir() if p.is_file()}


def prepare(project, output):
    parent = project / 'data/processed/regional_forest_age_inputs_v1'
    history = project / 'data/processed/regional_secondary_harvest_carbon_v1'
    if output.exists():
        raise ValueError('Refusing to overwrite an existing extension directory')
    for frozen in (parent, history):
        if output.resolve().is_relative_to(frozen.resolve()):
            raise ValueError('Extension must be outside both immutable parent bundles')
    frozen_stats = {str(p): snapshot(p) for p in (parent, history)}
    parent_sha, history_sha = digest(parent / 'manifest.json'), digest(history / 'manifest.json')
    pm = json.loads((parent / 'manifest.json').read_text())
    hm = json.loads((history / 'manifest.json').read_text())
    if hm['parent_input_manifest_sha256'] != parent_sha:
        raise ValueError('Historical extension belongs to a different parent input bundle')
    if (pm['source_config'] != hm['source_config'] or len(pm['source_config']) != 500
            or pm['ordered_source_ids_sha256'] != hm['ordered_source_ids_sha256']):
        raise ValueError('Original 500-member identity/order differs')
    projection_path = history / 'projection_parameters.nc'
    projection_sha = digest(projection_path)
    if projection_sha != hm['outputs']['projection_parameters.nc']['sha256']:
        raise ValueError('Frozen historical projection identity differs')
    projection = xr.load_dataset(projection_path)
    if (projection.source_config.values.tolist() != pm['source_config']
            or not np.array_equal(projection.config, np.arange(500))):
        raise ValueError('Projection member identity/order differs')
    prepared, scenarios = {}, {}
    for scenario, source_id in SOURCE_IDS.items():
        source_path = project / f'data/processed/land_use/future_luh3_r10_v3/{scenario}/land_use_absolute.nc'
        report_path = source_path.with_name('report.json')
        report = json.loads(report_path.read_text())
        schema = report['schema']['transitions']
        if (report['status'] != 'accepted_with_source_caveats' or report['scenario'] != scenario.upper()
                or schema['source_id'] != source_id or not schema['annual_bounds_checked']):
            raise ValueError('Unreviewed scenario source or time support')
        for label in SOURCES:
            if schema['variables'][label + '_bioh']['units'] != 'kg':
                raise ValueError('Original bioh units differ from grid-total kgC')
        receipts = report['source_receipts']
        for record in receipts:
            if (record['byte_validation'] != 'passed' or record['container_open'] != 'passed'
                    or record['container']['source_id'] != source_id
                    or record['container']['source_version'] != schema['source_version']
                    or not any('/CMIP7/ScenarioMIP/' in a['url'] for a in record['attempts'])):
                raise ValueError('Source acquisition receipt does not identify accepted CMIP7 data')
        with xr.open_dataset(source_path) as ds:
            for key in ('source_id', 'source_version', 'source_doi', 'scenario'):
                expected = scenario.upper() if key == 'scenario' else schema[key]
                if ds.attrs[key] != expected:
                    raise ValueError('Source cache/report identity mismatch: ' + key)
            source = ds[SOURCE_VARIABLES].sel(interval=slice(2022, 2098), harvest_source=SOURCES).load()
        forcing_name = f'future_{scenario}_forcing.nc'
        with xr.open_dataset(parent / forcing_name) as forcing:
            coordinates_equal(projection, forcing, ['config', 'source_config', 'reg_land', 'reg_land_name'])
            source, requests, diagnostic = prepare_source(source, forcing)
        requests.attrs.update(scenario='scen7-' + scenario.upper(), source_id=source_id,
                              source_doi=schema['source_doi'], data_status='candidate_data_only')
        prepared[scenario] = (source, requests)
        scenarios[scenario] = {
            'source_id': source_id, 'source_version': schema['source_version'], 'source_doi': schema['source_doi'],
            'iam_scenario': schema['iam_scenario'], 'original_bioh_metadata': {
                s + '_bioh': schema['variables'][s + '_bioh'] for s in SOURCES},
            'source_time': schema['time'], 'source_interval_archive': [2022, 2099],
            'source_selected_intervals': [2022, 2098], 'native_years': [2022, 2099],
            'processed_file': str(source_path), 'processed_file_sha256': report['output_sha256'],
            'processed_digest_basis': 'Accepted existing report; no repeated raw hashing',
            'processed_report': str(report_path), 'processed_report_sha256': digest(report_path),
            'original_source_receipts': [{k: r[k] for k in ('path', 'bytes', 'sha256', 'byte_validation', 'container_open', 'attempts')}
                                         for r in receipts],
            'parent_forcing': pm['outputs'][forcing_name],
            'parent_forcing_digest_basis': 'Frozen parent manifest',
            'source_carbon_policy': report['policy']['bioh_policy'],
            'source_bioh_correction_applied': report['bioh_correction_applied'],
            'source_plantation_caveat': schema['comment'],
            'source_H_warning': report['H_warning'], 'H_warning_applicable': scenario == 'h',
            'diagnostic': diagnostic,
        }
    # Everything scientific has validated before any output is created.
    output.mkdir(parents=True)
    outputs = {}
    for scenario, (source, requests) in prepared.items():
        for name, ds in [(f'source_{scenario}.nc', source), (f'future_{scenario}_requests.nc', requests)]:
            path = output / name
            ds.to_netcdf(path)
            with xr.open_dataset(path) as reopened:
                xr.testing.assert_equal(reopened.load(), ds)
            outputs[name] = {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest(path)}
    copied = output / projection_path.name
    shutil.copy2(projection_path, copied)
    if digest(copied) != projection_sha:
        raise RuntimeError('Projection copy changed bytes')
    outputs[copied.name] = {'path': str(copied), 'size_bytes': copied.stat().st_size, 'sha256': projection_sha,
                            'role': 'Unmodified historical projection byte copy; no new parameters'}
    if any(snapshot(Path(p)) != before for p, before in frozen_stats.items()):
        raise RuntimeError('An immutable parent bundle changed during preparation')
    if digest(parent / 'manifest.json') != parent_sha or digest(history / 'manifest.json') != history_sha:
        raise RuntimeError('An immutable parent manifest changed during preparation')
    receipt = {
        'schema': 'regional-secondary-harvest-carbon-future-v1', 'status': 'prepared_inputs_verified',
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'model_run': False,
        'scope': 'CMIP7 candidate data for the scenarios in this manifest; no future model integration or scientific selection',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        'preparer_sha256': digest(__file__),
        'parent_input_directory': str(parent), 'parent_input_manifest_sha256': parent_sha,
        'history_extension_directory': str(history), 'history_extension_manifest_sha256': history_sha,
        'parent_bundles_unchanged': True, 'parent_file_counts': {p: len(v) for p, v in frozen_stats.items()},
        'ordered_source_ids_sha256': pm['ordered_source_ids_sha256'], 'source_config': pm['source_config'],
        'projection': PROJECTION, 'projection_source_sha256': projection_sha, 'projection_byte_copy_identical': True,
        'contract': 'secondary_gross_agb_request_proportional', 'secondary_harvest_label': LABELS,
        'source_label_semantics': 'Upstream allocation labels, not independent observed young/mature donor pools',
        'event_scope': 'Repeated secondary secdf-to-secdf harvest only', 'time_contract': TIME,
        'units': 'PgC amount per complete annual interval',
        'conversion': 'Cached grid-total kgC reduced to regional PgC; no new area multiplication, slash division, HWP conversion, or clearing subtraction',
        'history_negative_policy': hm['source']['negative_cleanup_policy'],
        'negative_policy_author_guidance': hm['source']['official_guidance'],
        'extension_negative_policy': 'Apply the historical cell-negative-to-zero policy using existing regional negative_absolute; preserve signed and additive terms; no clipping of regional sums',
        'new_negative_policy': False, 'new_application_of_inherited_policy': True,
        'missing_policy': 'Reject missing, unavailable, negative corrected carbon or negative area; no imputation',
        'raw_rasters_read': False, 'raw_rehashed': False,
        'official_documentation': OFFICIAL_URL,
        'official_verification': 'The pinned official LUH3 documentation distinguishes biomass and area labels and retains an unresolved early-H allocation warning. Variable units follow the source-container metadata recorded in each source report.',
        'scenarios': scenarios, 'outputs': outputs,
        'limitations': ['Operational gross-AG working interpretation, not certified observed organ carbon',
                        'Same-cell positive-carbon/zero-area diagnostic was not retained; regional check passed only',
                        'Early-H official allocation warning retained; no repair or attribution of affected harvest',
                        'Source 2099 interval is outside the common native 2099 endpoint; not extrapolated or shifted'],
    }
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.project.resolve(), args.output.resolve())
    print(json.dumps({'status': result['status'], 'model_run': False,
                      'scenarios': {s: r['diagnostic'] for s, r in result['scenarios'].items()}}, indent=2))


if __name__ == '__main__':
    main()

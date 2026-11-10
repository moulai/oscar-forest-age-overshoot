"""Prepare pinned historical inputs and effective parameters without integrating OSCAR."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
sys.dont_write_bytecode = True
import numpy as np
import xarray as xr
from oscar._core._model.OSCAR import OSCAR
from oscar._core._base.cls_main import Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflows.cmip7_n2o_mass_basis import BASIS, STATIC, correct_legacy_fields, label_forcing, pi_identity

CONFIG = ROOT / 'config/reference_background.json'
REPORT = {'scope': 'Data preparation and native secondary-parameter evaluation only; no integration.',
          'checks': {}, 'outputs': [], 'failures': []}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def check(condition, label):
    if not condition:
        raise ValueError(label)


def exact_coords(a, b, dims):
    for dim in dims:
        check(np.array_equal(a[dim].values, b[dim].values), 'Coordinate mismatch: ' + dim)
    if 'config' in dims:
        check(np.array_equal(a.source_config, b.source_config), 'Source member IDs mismatch')


def equal_values(a, b):
    check(set(a.dims) == set(b.dims), 'Variable dimensions differ')
    a, b = xr.align(a, b, join='exact')
    return bool(np.array_equal(a.values, b.transpose(*a.dims).values, equal_nan=True))


def max_error(a, b):
    a, b = xr.broadcast(*xr.align(a, b, join='exact'))
    error = np.abs(a.values - b.transpose(*a.dims).values)
    check(np.isfinite(error).all(), 'Comparison contains nonfinite effective values')
    return float(error.max())


def inventory(ds):
    result = {'sizes': dict(ds.sizes), 'variables': {}}
    for name, var in ds.data_vars.items():
        v = var.values
        finite = np.isfinite(v)
        result['variables'][name] = {'dims': list(var.dims), 'dtype': str(var.dtype),
            'units': var.attrs.get('units'), 'finite': int(finite.sum()), 'nan': int(np.isnan(v).sum()),
            'positive_inf': int(np.isposinf(v).sum()), 'negative_inf': int(np.isneginf(v).sum()),
            'min': float(v[finite].min()) if finite.any() else None,
            'max': float(v[finite].max()) if finite.any() else None}
    return result


def save_nc(ds, path):
    check(not path.exists(), 'Refusing to replace output')
    partial = path.with_suffix('.nc.partial')
    ds.to_netcdf(partial, engine='netcdf4')
    with xr.open_dataset(partial) as reopened:
        xr.testing.assert_identical(ds, reopened.load())
    os.link(partial, path)
    partial.unlink()
    path.chmod(0o444)
    REPORT['outputs'].append({'path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size,
                              'sha256': sha(path), 'reopened_identical': True, **inventory(ds)})


def reject_model_call(*args, **kwargs):
    raise RuntimeError('Model integration is forbidden in input preparation')


def load_absolute(sources, cfg, halo_order):
    c = sources['concentrations']
    files = c['files']
    check(len(files) == 43 and len({v['species'] for v in files}) == 43, 'Gas files not unique')
    mapping = cfg['halo_mapping']
    check(len(mapping) == 40 and set(mapping.values()) == set(halo_order), 'Halo map does not match background')
    check(len(set(mapping.values())) == 40, 'Halo map not one-to-one')
    check(set(v['species'] for v in files) == set(mapping) | {'co2', 'ch4', 'n2o'}, 'Unexpected/missing gas')
    series = {}
    years = np.arange(1750, 2023)
    for item in files:
        path = ROOT / c['destination_directory'] / item['name']
        with xr.open_dataset(path) as ds:
            v = ds[item['species']]
            check(v.dims == ('time',) and v.size == 273, 'Gas time shape differs')
            check(v.attrs['units'] == item['units'], 'Gas units differ')
            check(v.attrs['cell_methods'] == 'area: time: mean', 'Gas averaging differs')
            check(ds.attrs['source_id'] == c['source_id'] and ds.attrs['source_version'] == c['source_version'], 'Gas source version differs')
            check(ds.attrs['grid_label'] == 'gm' and ds.attrs['frequency'] == 'yr', 'Gas grid or frequency differs')
            check(np.array_equal(ds.time.dt.year, years), 'Gas annual labels differ')
            bounds = ds[ds.time.attrs['bounds']].values
            starts = np.array([f'{y}-01-01' for y in years], dtype='datetime64[ns]')
            ends = np.array([f'{y+1}-01-01' for y in years], dtype='datetime64[ns]')
            check(np.array_equal(bounds[:, 0], starts) and np.array_equal(bounds[:, 1], ends), 'Annual CF bounds differ')
            check(np.isfinite(v.values).all(), 'Missing effective concentration')
            series[item['species']] = xr.DataArray(v.values.astype(np.float64), dims='year',
                coords={'year': years}, attrs={'units': item['units']})
    absolute = xr.Dataset({name.upper(): series[name] for name in ['co2', 'ch4', 'n2o']})
    reverse = {v: k for k, v in mapping.items()}
    absolute['Xhalo'] = xr.concat([series[reverse[k]] for k in halo_order],
        dim=xr.IndexVariable('spc_halo', halo_order), join='exact').transpose('year', 'spc_halo')
    absolute = absolute.expand_dims(scen=['hist'])
    absolute.attrs = {'source_id': c['source_id'], 'version': c['esgf_version'], 'doi': c['doi'],
        'license': 'CC BY 4.0', 'time_semantics': 'Integer year denotes full CF annual bounds; no date shift or interpolation'}
    return absolute


def prepare(cfg, sources, out):
    inputs = json.loads((ROOT / cfg['raw_acquisition_manifest']).read_text())['files']
    check(len(inputs) == 45 and len({v['path'] for v in inputs}) == 45, 'Expected 45 unique raw assets')
    for item in inputs:
        path = ROOT / item['path']
        check(path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256'], 'Raw input checksum mismatch')
    REPORT['raw_inputs'] = [{k: v[k] for k in ['path', 'size_bytes', 'sha256']} for v in inputs]
    REPORT['checks']['raw_input_hashes_before'] = 45
    with xr.open_dataset(ROOT / sources['parameters']['destination']) as ds:
        original = ds.load()
    with xr.open_dataset(ROOT / sources['forcing']['destination']) as ds:
        raw_forcing = correct_legacy_fields(ds.load())
    check(dict(original.sizes)['config'] == 500 and original.sizes['reg_land'] == 11, 'Parameter dimensions differ')
    check(original.sizes['bio_land'] == 5 and original.sizes['spc_halo'] == 53, 'Biome/halo dimensions differ')
    check(not set(original.data_vars) & OSCAR.var_param, 'Original parameters contain stale secondary cache')
    ids = original.config.values.copy()
    check(len(np.unique(ids)) == 500, 'Original IDs not unique')
    primary = original.assign_coords(config=np.arange(500), source_config=('config', ids))
    primary.source_config.attrs = {'long_name': 'Original ordered project ensemble member ID', 'source_coordinate': 'raw config'}
    primary.attrs = {'source': 'project-provided ensemble; generation method not provided',
        'source_sha256': sources['parameters']['sha256'], 'role': 'Original primary values with explicit internal/source member mapping'}
    for name in original.data_vars:
        check(primary[name].dtype == original[name].dtype and np.array_equal(primary[name].values, original[name].values, equal_nan=True), 'Primary value changed: ' + name)
    check(len(OSCAR.var_param) == 69, 'Native secondary parameter count changed')
    drop = cfg['remove_foreign_member_corrections']
    check({k for k, v in raw_forcing.data_vars.items() if 'config' in v.dims} == set(drop), 'Unexpected foreign-member variables')
    public = raw_forcing.drop_vars(drop + ['config']).sel(year=slice(1750, 2022)).drop_vars('scen')
    check('config' not in public.dims and 'source_config' not in public.coords, 'Foreign member axis remains')
    exact_coords(primary, public, ['reg_land', 'bio_land', 'bio_from', 'bio_to'])
    check(np.array_equal(public.year, np.arange(1750, 2023)), 'Background years differ')
    covered = public.spc_halo.values.tolist()
    absolute = load_absolute(sources, cfg, covered)
    p64 = primary.astype(np.float64)
    # The pure formatter imports native loader metadata; use the standard local path setup.
    from oscar._io import paths
    native_directory = ROOT / cfg['native_data_directory']
    if paths.SETTINGS_FILE.exists():
        check(paths.get_user_data_dir() == native_directory, 'Existing native data path differs; refusing to overwrite settings')
        settings_action = 'reused existing configuration'
    else:
        paths.set_data_dir(native_directory)
        settings_action = 'created ignored local configuration with native set_data_dir'
    REPORT['native_import_environment'] = {'settings': '.oscar_settings.json',
        'data_directory': cfg['native_data_directory'], 'action': settings_action, 'native_data_loaded': False}
    from oscar._core._base.fct_drivers import format_CMIP_conc
    pi = p64[['CO2_pi', 'CH4_pi', 'N2O_pi', 'Xhalo_pi']]
    formatted = format_CMIP_conc(absolute, conc_pi=pi, year_pi=1750, scen_hist='hist').sel(scen='hist', drop=True)
    conc = formatted[cfg['concentration_fields']]
    # Explicit 53-species alignment leaves missing 13 species as NaN, including emission placeholders.
    conc = conc.reindex(spc_halo=primary.spc_halo)
    public = public.astype(np.float64).reindex(spc_halo=primary.spc_halo)
    exact_coords(primary, conc, ['config', 'spc_halo'])
    exact_coords(public, conc, ['year', 'spc_halo'])
    forcing = xr.merge([public, conc], join='exact', compat='no_conflicts')
    forcing.attrs = {'role': 'Historical concentration-driven forcing with explicit background static references',
        'years': '1750-2022', 'dtype_contract': 'float64', 'coverage': '40 real halo species; 13 uncovered species remain NaN',
        'initialization': '1750 is native initialization; use prescribed D_CO2 from For, not zero ModelOut initialization'}
    label_forcing(forcing, public)
    check(len(forcing.data_vars) == 72 and not set(drop) & set(forcing.data_vars), 'Forcing variable set differs')
    check(set(forcing.data_vars) == set(public.data_vars) | set(cfg['concentration_fields']), 'Formatter reference fields leaked')
    check(not (set(OSCAR.var_in) - set(forcing.data_vars)), 'Native external inputs missing')
    uncovered = [v for v in primary.spc_halo.values.tolist() if v not in covered]
    coverage = {'species': primary.spc_halo.values.tolist(), 'covered': [v in covered for v in primary.spc_halo.values.tolist()],
        'mapping': cfg['halo_mapping'], 'covered_count': 40, 'uncovered_count': 13, 'uncovered_species': uncovered,
        'missing_value_policy': 'Leave uncovered concentrations, derivatives and halo emission placeholders as NaN; never fill missing history with zero'}
    for name in ['D_Xhalo', 'd_Xhalo', 'D_Eant_Xhalo']:
        check(np.isnan(forcing[name].sel(spc_halo=uncovered)).all(), 'Uncovered halo not masked: ' + name)
        check(np.isfinite(forcing[name].sel(spc_halo=covered)).all(), 'Covered halo not finite: ' + name)
    validation = {}
    for gas in ['CO2', 'CH4', 'N2O', 'Xhalo']:
        anomaly, derivative = forcing['D_' + gas], forcing['d_' + gas]
        reference = absolute[gas].sel(scen='hist', drop=True)
        param = p64[gas + '_pi']
        if gas == 'Xhalo':
            anomaly, derivative, param = [v.sel(spc_halo=covered) for v in [anomaly, derivative, param]]
        check(np.isfinite(anomaly).all() and np.isfinite(derivative).all(), 'Effective concentration fields not finite')
        check(anomaly.attrs['units'] == reference.attrs['units'] and derivative.attrs['units'] == reference.attrs['units'] + ' yr-1', 'Formatted concentration units differ')
        reconstruction_error = max_error(anomaly + param, reference)
        check(reconstruction_error < 1e-10, 'Concentration reconstruction differs')
        # An independent array-gradient check catches wrong axis, year shifts and endpoint handling.
        a = reference.transpose('year', ...)
        gradient = np.gradient(a.values, np.arange(1750, 2023), axis=0)
        gradient[0] = 0.
        expected = xr.DataArray(gradient, dims=a.dims, coords=a.coords)
        derivative_error = max_error(derivative, expected)
        check(derivative_error < 1e-10, 'Concentration derivative differs')
        check(bool((derivative.sel(year=1750) == 0).all()), 'Initial derivative not zero')
        end_error = max_error(derivative.sel(year=2022, drop=True), reference.sel(year=2022, drop=True) - reference.sel(year=2021, drop=True))
        check(end_error < 1e-10, 'Final one-sided derivative differs')
        validation[gas] = {'reconstruction_max_error': reconstruction_error, 'independent_derivative_max_error': derivative_error,
            'final_one_sided_error': end_error, 'units': anomaly.attrs['units'], 'derivative_units': derivative.attrs['units']}
    # Reject plausible member or regional misalignment before native outer joins can hide it.
    rejected = []
    for label, bad, dims in [('member_order', forcing.isel(config=slice(None, None, -1)), ['config']),
        ('source_member_ids', forcing.assign_coords(source_config=('config', ids[::-1])), ['config']),
        ('region_order', forcing.isel(reg_land=slice(None, None, -1)), ['reg_land'])]:
        try:
            exact_coords(primary, bad, dims)
        except ValueError:
            rejected.append(label)
    check(len(rejected) == 3, 'Misalignment probe was accepted')
    static = [k for k, v in forcing.data_vars.items() if 'year' not in v.dims]
    check(len(static) == 30, 'Static field count differs')
    exact_coords(p64, forcing, ['config', 'reg_land', 'bio_land', 'bio_from', 'bio_to', 'spc_halo'])
    moved, time_forcing = OSCAR._move_For_to_Par(p64.copy(deep=True), forcing.copy(deep=True))
    provenance, changed = {}, []
    for name in static:
        exists = name in p64
        equal = equal_values(*xr.broadcast(p64[name], forcing[name])) if exists else None
        if exists and not equal:
            changed.append(name)
        provenance[name] = {'source': 'CMIP7/IAMC_R10 background static reference' + ('; documented two-N-atom mass correction' if name in STATIC else ''),
            'source_variable': name, 'existed_in_primary': exists, 'equal_after_broadcast': equal,
            'primary_dims': list(p64[name].dims) if exists else None, 'effective_dims': list(moved[name].dims)}
        check(equal_values(moved[name], forcing[name]), 'Native static move differs: ' + name)
    check(set(changed) == set(cfg['changed_static_fields']) and len(changed) == 19, 'Static override changed unexpected values')
    for name in cfg['unchanged_land_static_fields']:
        check(provenance[name]['equal_after_broadcast'], 'Land static reference differs')
    for name in ['Eant_H2_pi', 'Ebb_H2_pi']:
        check(bool((moved[name] == 0).all()), 'H2 emission control not zero')
    for name in ['D_Eant_H2', 'D_Ebb_H2', 'D_H2', 'd_H2']:
        check(bool((forcing[name] == 0).all()), 'H2 time control not zero')
    check(bool((moved.H2_pi == moved.H2_pd).all()), 'H2 PI/PD controls differ')
    secondary = OSCAR._get_Par2(moved.copy(deep=True))
    check(set(secondary.data_vars) == OSCAR.var_param, 'Incomplete secondary parameter result')
    effective = xr.merge([secondary, moved], join='exact', compat='no_conflicts')
    effective.attrs = {'role': 'Audited float64 effective parameter snapshot, after native static move and native secondary evaluation',
        'input_role': 'Audit snapshot; later runner must rebuild from indexed primary plus forcing and must not reuse stale secondary caches'}
    label_forcing(effective, effective)
    untouched = [k for k in p64.data_vars if k not in static]
    for name in untouched:
        check(equal_values(effective[name], p64[name]), 'Unauthorized primary change: ' + name)
    for name in static:
        check(equal_values(effective[name], forcing[name]), 'Effective static reference differs: ' + name)
    # Recalculate the original OSCAR reference independently to record secondary-parameter differences.
    old_secondary = OSCAR._get_Par2(p64.copy(deep=True))
    secondary_provenance = {k: {'source': 'explicit background static override' if k in static else 'native secondary equation',
        'changed_from_original_reference': not equal_values(*xr.broadcast(secondary[k], old_secondary[k]))} for k in sorted(OSCAR.var_param)}
    check(bool((effective.Ebb_H2_pi == 0).all()), 'Explicit H2 secondary override lost')
    check(all(v.dtype == np.dtype('float64') for v in effective.data_vars.values()), 'Effective dtype differs')
    exact_coords(primary, effective, ['config', 'reg_land', 'bio_land', 'spc_halo'])
    REPORT.update(n2o_mass_basis=BASIS, n2o_pi_sha256=pi_identity(effective),
        n2o_source_correction='Four identified v4-beta2.1 archive fields multiplied by two before native preparation; original parameters and raw bytes unchanged',
        static_provenance=provenance, secondary_provenance=secondary_provenance,
        secondary_count=69, secondary_recomputed_count=68, secondary_explicit_override_count=1,
        unchanged_primary_variable_count=len(untouched), coverage=coverage, concentration_checks=validation,
        source_config=ids.tolist(), future_run_contract=cfg['future_run_contract'],
        initial_D_CO2_range=[float(forcing.D_CO2.sel(year=1750).min()), float(forcing.D_CO2.sel(year=1750).max())])
    REPORT['checks'].update(raw_primary_values_and_dtypes_preserved=True, all_members_preserved=True,
        exact_coordinates=True, rejected_misalignment_cases=rejected, no_foreign_emission_corrections=True,
        only_eight_formatter_fields_merged=True, H2_disabled=True, no_stale_secondary_cache=True,
        static_changed_count=19, native_required_external_inputs_present=True)
    for name, ds in [('indexed_primary_parameters.nc', primary), ('historical_concentration_forcing.nc', forcing),
                     ('effective_parameters.nc', effective), ('absolute_concentrations.nc', absolute)]:
        save_nc(ds, out / name)
    write_json(out / 'coverage.json', coverage)
    REPORT['coverage_file'] = {'path': str((out / 'coverage.json').relative_to(ROOT)), 'sha256': sha(out / 'coverage.json')}
    for item in inputs:
        check(sha(ROOT / item['path']) == item['sha256'], 'Raw file changed during preparation')
    REPORT['checks']['raw_input_hashes_after'] = 45


def main():
    cfg = json.loads(CONFIG.read_text())
    check(cfg['years'] == [1750, 2022], 'Unsupported preparation years')
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD:oscar'], cwd=ROOT, text=True).strip()
    check(tree == cfg['expected_oscar_tree'], 'Core tree differs from preparation contract')
    check(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip(), 'Commit preparation code before execution')
    out = ROOT / cfg['output_directory']
    out.mkdir(parents=True, exist_ok=False)
    REPORT.update(started_utc=now(), code_commit=code, oscar_tree=tree, config_sha256=sha(CONFIG),
                  command='.venv/bin/python -B workflows/prepare_reference_background.py',
                  source_manifest_sha256=sha(ROOT / cfg['raw_source_manifest']))
    try:
        with patch.object(Model, '__call__', reject_model_call):
            prepare(cfg, json.loads((ROOT / cfg['raw_source_manifest']).read_text()), out)
        REPORT['status'] = 'prepared_bytes_and_contract_checks_passed'
    except Exception as exc:
        REPORT['status'] = 'failed_at_preparation_node'
        REPORT['failures'].append({'type': type(exc).__name__, 'message': str(exc).replace(str(ROOT), '<project>')})
        raise
    finally:
        REPORT['finished_utc'] = now()
        write_json(out / 'manifest.json', REPORT)
        print(REPORT['status'], str(out.relative_to(ROOT)), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Fixed-ensemble algebraic preflight; never calls Model.__call__."""
import hashlib
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CFG_PATH = ROOT / 'config/growth_reference.json'
CFG = json.loads(CFG_PATH.read_text())
OUT = ROOT / CFG['output_directory']
DIMS = ('reference_environment', 'reg_land', 'bio_land', 'config')
RATE = 'yr-1'
FLUX = 'PgC Mha-1 yr-1'
BASE = ['npp_pi', 'v_mort', 'v_fire', 'p_charv', 'p_graz', 'p_wood', 't_wood']
DEPS = {
    'r_npp': ['CO2_pi', 'b2_npp_CO2', 'x_npp_CO2', 'g_npp_T', 'g_npp_T2', 'x_npp_P'],
    'r_vmort': ['v_mort', 'x_mort_npp', 'g_mort_T', 'x_mort_P'],
    'r_vfire': ['v_fire', 'x_fire_npp', 'x_fire_CO2', 'g_fire_T', 'g_fire_P', 'Pl_pi', 'CO2_pi'],
}

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')

def finite_summary(a):
    v = np.asarray(a)
    f = v[np.isfinite(v)]
    return {'size': int(v.size), 'finite': int(f.size), 'nan': int(np.isnan(v).sum()),
            'positive_inf': int(np.isposinf(v).sum()), 'negative_inf': int(np.isneginf(v).sum()),
            'min': float(f.min()) if f.size else None, 'max': float(f.max()) if f.size else None}

def main():
    if OUT.exists():
        raise FileExistsError(f'No-clobber: {OUT}')
    code_commit = git('rev-parse', 'HEAD')
    for name in ['workflows/prepare_growth_reference.py', 'config/growth_reference.json', 'docs/methods.md']:
        if git('status', '--porcelain', '--', name):
            raise RuntimeError(f'Commit workflow contract before execution: {name}')
    assert git('rev-parse', 'HEAD:oscar') == CFG['oscar_tree']
    assert not git('status', '--porcelain', '--', 'oscar')
    acceptance_path = ROOT / CFG['acceptance_path']
    assert sha(acceptance_path) == CFG['acceptance_sha256']
    accepted = json.loads(acceptance_path.read_text())
    assert accepted['status'] == 'accepted_frozen'
    common_info = accepted['artifacts']['common_environment.nc']
    assert common_info['sha256'] == CFG['common_environment_sha256']
    prep_path = ROOT / accepted['input_preparation']['manifest_path']
    assert sha(prep_path) == accepted['input_preparation']['manifest_sha256']
    prep = json.loads(prep_path.read_text())
    inputs = {CFG['acceptance_path']: CFG['acceptance_sha256'],
              common_info['path']: common_info['sha256'],
              accepted['input_preparation']['manifest_path']: accepted['input_preparation']['manifest_sha256']}
    needed = ['indexed_primary_parameters.nc', 'effective_parameters.nc', 'historical_concentration_forcing.nc']
    for info in prep['outputs']:
        if Path(info['path']).name in needed:
            inputs[info['path']] = info['sha256']
    assert len(inputs) == 6
    for path, digest in inputs.items():
        assert sha(ROOT / path) == digest, path
    P, E, F = [xr.load_dataset(prep_path.parent / n) for n in needed]
    H = xr.load_dataset(ROOT / common_info['path'])
    assert H.sizes['config'] == CFG['member_count'] == 500
    assert H.config.values.tolist() == list(range(500))
    assert sha_ids(H.source_config) == accepted['member_mapping']['ordered_source_ids_sha256']
    for name in ['config', 'source_config', 'reg_land']:
        for ds in [P, E, F]:
            np.testing.assert_array_equal(ds[name], H[name])
    for ds in [P, E, F]:
        assert ds.bio_land.values.tolist() == accepted['coordinates']['bio_land']
    assert H.reg_land.values.tolist() == accepted['coordinates']['reg_land']
    np.testing.assert_array_equal(H.D_CO2, F.D_CO2.transpose(*H.D_CO2.dims))
    expected_units = {'D_CO2': 'ppm', 'D_Tl': 'K', 'r_Pl': '1'}
    for name, unit in expected_units.items():
        assert H[name].attrs['units'] == unit
    for name, unit in {'npp_pi': FLUX, 'v_mort': RATE, 'v_fire': RATE, 'p_wood': '1',
                       't_wood': 'yr', 'p_charv': '1', 'p_graz': '1'}.items():
        assert E[name].attrs['units'] == unit, name
    frames = []
    for spec in CFG['reference_environments']:
        if 'start_year' in spec:
            years = np.arange(spec['start_year'], spec['end_year'] + 1)
            part = H.sel(year=years)
            assert len(years) == 20
            assert not part.initialization_record.any()
            d = part[list(expected_units)].mean('year', skipna=False, keep_attrs=True)
            for n in expected_units:
                d[n + '_annual_count'] = part[n].count('year').astype('int16')
                assert bool((d[n + '_annual_count'] == 20).all())
            d['start_year'], d['end_year'], d['year_count'] = spec['start_year'], spec['end_year'], 20
        else:
            d = H[list(expected_units)].isel(year=0, drop=True).copy()
            for n in expected_units:
                d[n] = xr.full_like(d[n], spec[n], dtype='float64')
                d[n + '_annual_count'] = xr.zeros_like(d[n], dtype='int16')
            d['start_year'], d['end_year'], d['year_count'] = -1, -1, 0
        d['initialization_records_used'] = 0
        frames.append(d.expand_dims(reference_environment=[spec['id']]))
    env = xr.concat(frames, dim='reference_environment', join='exact')
    for n, unit in expected_units.items():
        env[n].attrs.update(units=unit, meaning='Mean input before nonlinear response; parameter_PI is prescribed')
        env[n + '_annual_count'].attrs.update(units='1', meaning='Finite annual source records; parameter_PI has no observed records')
    for n in ['start_year', 'end_year', 'year_count', 'initialization_records_used']:
        env[n].attrs['units'] = 'yr' if n in ['start_year', 'end_year'] else '1'
    env.attrs.update(aggregation='Arithmetic mean of 20 input years, skipna=False, before nonlinear response',
                     area_support='Not assessed; no LUH area input or area weighting',
                     parameter_PI='D_CO2=0, D_Tl=0, r_Pl=1 per member; not observed 1750 anomaly',
                     code_commit=code_commit, common_environment_sha256=common_info['sha256'])
    native = importlib.import_module('oscar._core._model._modules.OSCAR_landC_density')
    model = native.OSCAR_landC_density
    var = env[list(expected_units)].copy()
    out = xr.Dataset()
    native_checks = {}
    for n in ['r_npp', 'r_vmort', 'r_vfire']:
        var[n] = model[n](var, E)
        direct = getattr(native, 'Eq__' + n)(var, E)
        xr.testing.assert_equal(var[n], direct)
        out[n] = var[n].transpose(*DIMS)
        out[n].attrs['units'] = model[n].units
        native_checks[n] = 'Exact native Process/Eq equality including nonfinite masks'
    out['npp'] = E.npp_pi * out.r_npp
    out['input_fraction'] = 1 - E.p_charv - E.p_graz
    out['npp_eff'] = out.npp * out.input_fraction
    out['mort_rate'] = E.v_mort * out.r_vmort
    out['fire_rate'] = E.v_fire * out.r_vfire
    out['lambda_loss'] = out.mort_rate + out.fire_rate
    for n in BASE:
        out['parameter_' + n] = E[n]
    out['wood_channel_rate'] = E.p_wood / E.t_wood
    degenerate = (E.v_mort == 0) & (out.wood_channel_rate == 0)
    out['degenerate_zero_mortality_channel'] = degenerate.astype('int8')
    out['beta_cwd'] = (out.wood_channel_rate / E.v_mort).where(E.v_mort != 0).where(~degenerate, 0.)
    out['legacy_fixed_wood_residual_rate'] = out.mort_rate - out.wood_channel_rate
    out['analytical_reference_valid'] = (np.isfinite(out.npp_eff) & (out.npp_eff >= 0) &
                                         np.isfinite(out.lambda_loss) & (out.lambda_loss > 0)).astype('int8')
    out['analytical_cveg_reference'] = (out.npp_eff / out.lambda_loss).where(out.analytical_reference_valid == 1)
    for n in ['npp', 'npp_eff']:
        out[n].attrs['units'] = FLUX
    for n in ['mort_rate', 'fire_rate', 'lambda_loss', 'wood_channel_rate', 'legacy_fixed_wood_residual_rate']:
        out[n].attrs['units'] = RATE
    out.analytical_cveg_reference.attrs.update(units='PgC Mha-1',
        meaning='Algebraic npp_eff/lambda only; not dynamic spin-up, initialized stock, or validated C_infinity')
    for n in ['input_fraction', 'beta_cwd', 'degenerate_zero_mortality_channel', 'analytical_reference_valid']:
        out[n].attrs['units'] = '1'
    template = xr.ones_like(out.r_npp, dtype=bool)
    masks = {}
    for n in ['r_npp', 'r_vmort', 'r_vfire']:
        mask = template.copy()
        for p in DEPS[n]:
            mask = mask & np.isfinite(E[p])
        for v in (['D_CO2', 'D_Tl', 'r_Pl'] if n != 'r_vmort' else ['D_Tl', 'r_Pl']):
            mask = mask & np.isfinite(env[v])
        if n != 'r_npp':
            mask = mask & masks['r_npp'] & np.isfinite(out.r_npp)
        masks[n] = mask
        out[n + '_source_finite'] = mask.astype('int8')
        out[n + '_output_finite'] = np.isfinite(out[n]).astype('int8')
        for suffix in ['_source_finite', '_output_finite']:
            out[n + suffix].attrs.update(units='1', meaning='Finite arithmetic support only; not LUH area or ecological support')
    issues = []
    def check(name, bad, message, blocker=True):
        bad = bad.transpose(*[d for d in DIMS if d in bad.dims])
        flag = 'failure_' + name
        out[flag] = bad.astype('int8')
        out[flag].attrs.update(units='1', meaning=message, blocks_acceptance=int(blocker))
        count = int(bad.sum())
        fcount = int(bad.sel(bio_land='Forest').sum()) if 'bio_land' in bad.dims else count
        examples = []
        for idx in np.argwhere(bad.values)[:12]:
            row = {d: (bad[d].values[i].item() if hasattr(bad[d].values[i], 'item') else bad[d].values[i]) for d, i in zip(bad.dims, idx)}
            if 'config' in row:
                row['source_config'] = int(H.source_config.sel(config=row['config']))
            examples.append(row)
        issues.append({'name': name, 'count': count, 'Forest_count': fcount,
                       'blocker': blocker, 'meaning': message, 'examples': examples})
    for n in BASE:
        a = E[n]
        forest = a.sel(bio_land='Forest') if 'bio_land' in a.dims else a
        check(n + '_Forest_nonfinite', ~np.isfinite(forest), 'Nonfinite Forest physical parameter')
        check(n + '_negative', np.isfinite(a) & (a < 0), 'Negative physical parameter in any retained biome')
    check('t_wood_nonpositive', np.isfinite(E.t_wood) & (E.t_wood <= 0), 'Wood timescale must be positive')
    check('p_wood_above_one', np.isfinite(E.p_wood) & (E.p_wood > 1), 'Stock fraction above one')
    check('p_wood_equal_one', E.p_wood == 1, 'Boundary: nonwood recovery decomposition is undefined at p_wood=1', False)
    for n in ['p_charv', 'p_graz']:
        check(n + '_above_one', np.isfinite(E[n]) & (E[n] > 1), 'Removal fraction above one')
        check(n + '_Forest_nonzero', E[n].sel(bio_land='Forest') != 0, 'Expected Forest removal fraction is zero')
    check('input_fraction_negative', np.isfinite(out.input_fraction) & (out.input_fraction < 0), 'Negative effective input fraction')
    check('input_fraction_Forest_nonpositive', out.input_fraction.sel(bio_land='Forest') <= 0, 'Forest effective input fraction must be positive')
    check('beta_invalid', (np.isfinite(out.beta_cwd) & ((out.beta_cwd < 0) | (out.beta_cwd > 1))) |
          ((E.v_mort == 0) & (out.wood_channel_rate > 0)), 'Baseline mortality partition outside [0,1] or positive wood loss with zero mortality')
    check('beta_Forest_nonfinite', ~np.isfinite(out.beta_cwd.sel(bio_land='Forest')), 'Forest beta must be defined, including explicit zero channel convention')
    for n in ['r_npp', 'r_vmort', 'r_vfire', 'npp', 'npp_eff', 'mort_rate', 'fire_rate', 'lambda_loss']:
        check(n + '_negative', np.isfinite(out[n]) & (out[n] < 0), 'Negative native response/input/rate')
        check(n + '_Forest_nonfinite', ~np.isfinite(out[n].sel(bio_land='Forest')), 'Nonfinite Forest response/input/rate')
    check('lambda_Forest_nonpositive', out.lambda_loss.sel(bio_land='Forest') <= 0, 'Forest total mature loss rate must be positive')
    check('legacy_fixed_wood_negative', np.isfinite(out.legacy_fixed_wood_residual_rate) &
          (out.legacy_fixed_wood_residual_rate < 0), 'Diagnostic: native fixed wood channel exceeds modern mortality; future beta split differs', False)
    pi_checks = {}
    for n in ['r_npp', 'r_vmort', 'r_vfire']:
        pi = out[n].sel(reference_environment='parameter_PI')
        support = masks[n].sel(reference_environment='parameter_PI')
        check(n + '_PI_response', support & (~np.isfinite(pi) | (abs(pi - 1) > CFG['pi_response_atol'])),
              'PI response must be one on finite source support')
        pi_checks[n] = finite_summary((pi - 1).where(support))
    blocked = [i for i in issues if i['blocker'] and i['count']]
    status = 'BLOCKED_PHYSICAL_INPUT' if blocked else 'PREFLIGHT_PASS_PENDING_INDEPENDENT_REVIEW'
    out.attrs.update(status=status, code_commit=code_commit, scope=CFG['primary_scope'],
        input_source='Accepted effective_parameters.nc, unchanged full 500-member snapshot',
        mortality_input='Only native environmental r_npp; no age multiplier',
        area_support='Unknown; zero or missing channels do not certify zero LUH area',
        beta_scope='Diagnostic proposed conservative baseline split; not implemented routing',
        missingness='Native NaN/Inf preserved; no fill or member filtering',
        analytical_reference='Not spin-up and not proof of actual C_infinity', model_calls=0)
    env.attrs['status'] = status
    for path, digest in inputs.items():
        assert sha(ROOT / path) == digest, path
    OUT.mkdir(parents=True, exist_ok=False)
    for name, ds in [('reference_environments.nc', env), ('mature_process_rates.nc', out)]:
        ds.to_netcdf(OUT / name)
        xr.testing.assert_identical(ds, xr.load_dataset(OUT / name))
    summary = {
        'status': status, 'blocked_checks': blocked, 'all_checks': issues,
        'sources_hashed_before_and_after': len(inputs), 'exact_shared_coordinates': True,
        'ordered_source_ids_sha256': sha_ids(H.source_config), 'members': 500,
        'modern_year_counts': [20, 20], 'initialization_rows_used': 0,
        'native_Process_Eq_checks': native_checks, 'PI_response_deviation': pi_checks,
        'netcdf_roundtrip_identical': True, 'model_calls': 0,
        'source_parameter_finite_summary': {n: finite_summary(E[n]) for n in sorted(set(BASE + sum(DEPS.values(), [])))},
        'source_masks': {n: {'finite_support': int(masks[n].sum()),
                            'nonfinite_output_on_finite_source_support': int((masks[n] & ~np.isfinite(out[n])).sum()),
                            'nonfinite_output_on_missing_source_support': int((~masks[n] & ~np.isfinite(out[n])).sum())}
                         for n in masks},
        'biome_summary': {str(b): {n: finite_summary(out[n].sel(bio_land=b)) for n in
                          ['r_npp', 'r_vmort', 'r_vfire', 'npp', 'npp_eff', 'mort_rate', 'fire_rate', 'lambda_loss', 'beta_cwd']}
                          for b in E.bio_land.values},
        'zero_mortality_channels': int(degenerate.sum()), 'p_wood_one_channels': int((E.p_wood == 1).sum()),
        'area_support': 'Not assessed; no missing/zero value excused using assumed zero area',
    }
    dump(OUT / 'checks.json', summary)
    outputs = [{ 'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'size_bytes': p.stat().st_size }
               for p in [OUT/'reference_environments.nc', OUT/'mature_process_rates.nc', OUT/'checks.json']]
    code_paths = ['workflows/prepare_growth_reference.py', 'config/growth_reference.json', 'docs/methods.md',
                  'oscar/_core/_model/_modules/OSCAR_landC_density.py', 'oscar/_core/_base/fct_solve.py', 'oscar/_core/_base/cls_main.py']
    manifest = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': status, 'code_commit': code_commit, 'oscar_tree': CFG['oscar_tree'],
        'python': sys.version, 'numpy': np.__version__, 'xarray': xr.__version__,
        'inputs': inputs, 'code_files': {p: sha(ROOT/p) for p in code_paths}, 'outputs': outputs,
        'member_mapping': {'config': H.config.values.tolist(), 'source_config': H.source_config.values.tolist()},
        'regions': E.reg_land.values.tolist(), 'biomes': E.bio_land.values.tolist(),
        'references': CFG['reference_environments'], 'source_dependency_dictionary': DEPS,
        'variable_dictionary': {name: {v: {'dims': list(ds[v].dims), 'dtype': str(ds[v].dtype), 'attrs': ds[v].attrs}
                              for v in ds.data_vars}
                              for name,ds in [('reference_environments.nc', env), ('mature_process_rates.nc', out)]},
        'physical_blocker_names': [b['name'] for b in blocked], 'excluded': CFG['excluded'],
        'scope': 'Algebraic diagnostic artifact; dynamic and ecological performance are not evaluated'}
    dump(OUT / 'manifest.json', manifest)
    print(json.dumps({'status': status, 'output_directory': str(OUT),
                      'physical_blockers': [{k: b[k] for k in ['name','count','Forest_count']} for b in blocked]}))
    return 2 if blocked else 0

def sha_ids(a):
    return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()

if __name__ == '__main__':
    sys.exit(main())

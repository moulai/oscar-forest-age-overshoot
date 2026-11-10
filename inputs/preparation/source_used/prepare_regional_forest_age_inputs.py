#!/usr/bin/env python3
"""Prepare labeled native forest-age inputs from accepted processed sources.

No model integration, resampling, grid reconstruction or parameter fitting is
performed here. Secondary parameters are rebuilt by the native model adapter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEGACY_LAND_STATIC = ('Aland_bk_pi', 'dC_wharv_pi', 'dA_wharv_pi', 'dA_shift_pi', 'age_bk_pi')
LEGACY_LAND_DYNAMIC = ('D_dA_wharv1', 'D_dA_wharv2', 'D_dA_lcc1', 'D_dA_lcc2', 'D_dA_shift')
ENVIRONMENT = ('D_CO2', 'D_Tl', 'r_Pl')
SEMANTICS = ('event_source_class', 'event_target_class', 'event_kind',
             'event_vegetation_action', 'event_age_action', 'event_retained_split',
             'event_evidence_status', 'class_biome', 'class_role')
MEMBER_HASH = 'ae263948f561c541cb3f6fcec5d8f2cc399b51c108b4613be61c626b598df81e'
SOURCES = {
    'primary': 'data/processed/reference_background/v1/indexed_primary_parameters.nc',
    'historical_atmosphere': 'data/processed/reference_background/v1/historical_concentration_forcing.nc',
    'future_atmosphere': 'data/processed/future_background/v3/atmospheric_forcing.nc',
    'calibration': 'data/processed/bookkeeping/main-candidate16-v1/inputs/calibration_environment.nc',
    'pi': 'data/processed/bookkeeping/main-candidate16-v1/inputs/pi_environment.nc',
    'common_environment': 'results/science_baseline/ensemble-20261103T204649Z-9a8a68c3/common_environment.nc',
    'growth_curve': 'data/processed/regional_continuous_curves/v1/hermite/regional_continuous_curves.nc',
    'wood_calibration': 'data/processed/regional_continuous_curves/v1/hermite/wood_calibration.nc',
}


def exact_coordinates(primary, other, dimensions=('config', 'reg_land')):
    """Reject reordering or relabeling instead of silently joining members."""
    for dim in dimensions:
        if dim in other.dims and not np.array_equal(primary[dim], other[dim]):
            raise ValueError('Coordinate mismatch: ' + dim)
    if 'config' in other.dims:
        if 'source_config' not in other.coords or not np.array_equal(primary.source_config, other.source_config):
            raise ValueError('Original member identity/order mismatch')


def initial_biome_area(land, biomes):
    area = land.Aland_initial
    if not np.all(np.isfinite(area)) or bool((area < 0).any()):
        raise ValueError('Invalid initial area')
    if bool((area.where(land.class_role == 'forest_cohort', 0) != 0).any()):
        raise ValueError('Nonzero initial cohort area needs an explicit age history')
    return xr.concat([area.where(land.class_biome == b, 0).sum('luh_class') for b in biomes],
                     dim=xr.IndexVariable('bio_land', biomes)).transpose('reg_land', 'bio_land')


def prepare_primary(primary, atmosphere, land):
    """Apply documented static forcing references and the actual 850 land area."""
    exact_coordinates(primary, atmosphere)
    exact_coordinates(primary, land, ('reg_land',))
    out = primary.astype(np.float64).copy(deep=True)
    static = [name for name, value in atmosphere.data_vars.items() if 'year' not in value.dims]
    for name in static:
        if name not in LEGACY_LAND_STATIC and name != 'Aland_pi':
            out[name] = atmosphere[name]
    out = out.drop_vars([n for n in LEGACY_LAND_STATIC if n in out])
    out['Aland_pi'] = initial_biome_area(land, out.bio_land.values.tolist())
    out.Aland_pi.attrs = {'units': 'Mha', 'source': 'Regional raw-positive 850 initial area; no normalization'}
    out.attrs = {'role': 'Original ordered primary ensemble with explicit atmospheric static references and actual 850 area',
                 'secondary_parameters': 'None cached; rebuild with the native forest-age adapter',
                 'Aland_pd_policy': 'Retain the original native present-day reference definition',
                 'stock_baseline_policy': 'Install new reference and fire baselines only after dynamic PI spin-up'}
    return out, static


def regional_forcing(land, start, end, atmosphere=None):
    """Right-end annual intervals; the first row is always a restart anchor."""
    sliced = land.sel(time=slice(start, end))
    expected = np.arange(start, end + 1)
    if not np.array_equal(sliced.time, expected):
        raise ValueError('Land time coverage is not complete')
    events = sliced[['event_area']].rename(time='year').copy(deep=True)
    if events.event_area.attrs.get('units') != 'Mha':
        raise ValueError('Expected annual event areas in Mha')
    events.event_area[0] = 0.
    if atmosphere is None:
        result = events.assign(D_CO2=xr.zeros_like(events.year, dtype=float),
                               D_Tl=xr.zeros_like(events.year, dtype=float),
                               r_Pl=xr.ones_like(events.year, dtype=float))
        mode = 'Prescribed parameter-PI environment; regional land component only'
    else:
        source = atmosphere.sel(year=slice(start, end))
        if not np.array_equal(source.year, expected):
            raise ValueError('Atmospheric time coverage is not complete')
        if not np.array_equal(source.reg_land, events.reg_land):
            raise ValueError('Atmospheric and land region order differs')
        names = [n for n, v in source.data_vars.items() if 'year' in v.dims and n not in LEGACY_LAND_DYNAMIC]
        result = xr.merge([events, source[names]], join='exact', compat='no_conflicts')
        mode = 'Native concentration-driven atmosphere and climate; no prescribed climate trajectory'
    result.attrs = {'schema': 'regional-forest-age-native-forcing-v1', 'environment_mode': mode,
                    'time_contract': 'First row is an initial/restart anchor; later label y represents (y-1,y], event at y-0.5',
                    'event_anchor': 'Zeroed deliberately; any activity ending at the anchor belongs to the preceding phase',
                    'land_parent_identity': land.attrs['parent_identity'],
                    'legacy_land_drivers': 'Excluded; event_area is the sole land activity driver',
                    'atmospheric_coverage': 'Preserve source coverage, including unavailable halo species as NaN'}
    return result


def layout_from_inputs(parameters, metadata, births):
    """Construct model metadata without reading source rasters or changing IDs."""
    from oscar._core._model._modules.land_geometry import LandClass, EventSpec
    from oscar._core._model._modules.regional_forest_age import RegionalLayout
    classes = tuple(LandClass(str(name), str(b), str(role)) for name, b, role in
                    zip(metadata.luh_class.values, metadata.class_biome.values, metadata.class_role.values))
    events = tuple(EventSpec(str(name), str(metadata.event_source_class.values[i]),
                            str(metadata.event_target_class.values[i]), str(metadata.event_kind.values[i]),
                            str(metadata.event_vegetation_action.values[i]), str(metadata.event_age_action.values[i]),
                            'proportional', str(metadata.event_retained_split.values[i]))
                   for i, name in enumerate(metadata.event.values))
    return RegionalLayout(tuple(parameters.reg_land.values), tuple(parameters.config.values), classes,
                          tuple(births), tuple(parameters.bio_land.values), tuple(parameters.box_hwp.values),
                          events, source_members=tuple(parameters.source_config.values))


def load_runtime_inputs(directory, *, births, tau_years, age_enabled=True, weighting='capacity', member_slice=None):
    """Load prepared inputs; no integration or implicit initial stock is supplied.

    Experiments must name tau_years explicitly; no production choice is inferred.
    """
    from oscar._core._model._modules.regional_forest_age_reference import prepare_native_forest_parameters
    from oscar._core._model._modules.growth_curve import load_growth_curve
    from oscar._core._model._modules.land_carbon import AgeProduction
    directory = Path(directory)
    primary = xr.load_dataset(directory / 'primary_parameters.nc')
    calibration = xr.load_dataset(directory / 'calibration_environment.nc')
    if member_slice is not None:
        if (not isinstance(member_slice, slice) or member_slice.step not in (None, 1)
                or member_slice.start is None or member_slice.stop is None
                or not 0 <= member_slice.start < member_slice.stop <= primary.sizes['config']):
            raise ValueError('Computational members must be one bounded contiguous original slice')
        primary = primary.isel(config=member_slice)
        calibration = calibration.isel(config=member_slice)
    parameters = prepare_native_forest_parameters(primary, calibration)
    metadata = xr.load_dataset(directory / 'land_metadata.nc')
    layout = layout_from_inputs(parameters, metadata, births)
    kernels = None
    if age_enabled:
        curve = load_growth_curve(directory / 'growth_curve.nc', weighting=weighting)
        wood = xr.load_dataset(directory / 'wood_calibration.nc')
        if member_slice is not None:
            wood = wood.isel(config=member_slice)
        exact_coordinates(primary, wood)
        kernels = tuple(AgeProduction.from_calibration(curve, wood, r, tau_years=tau_years)
                        for r in layout.regions)
    return parameters, layout, kernels, metadata.Aland_initial


def fingerprint(path):
    path = Path(path)
    return {'path': str(path), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(data_root, output):
    data_root, output = Path(data_root), Path(output)
    if output.exists():
        raise FileExistsError('Refusing to replace prepared inputs')
    data = {name: xr.load_dataset(data_root / path) for name, path in SOURCES.items()
            if name != 'common_environment'}
    primary = data['primary']
    digest = hashlib.sha256(np.asarray(primary.source_config, dtype='<i8').tobytes()).hexdigest()
    if primary.sizes['config'] != 500 or digest != MEMBER_HASH or not np.array_equal(primary.config, np.arange(500)):
        raise ValueError('Original ordered 500-member ensemble required')
    for name in ('historical_atmosphere', 'future_atmosphere', 'calibration', 'pi', 'wood_calibration'):
        exact_coordinates(primary, data[name])
    exact_coordinates(primary, data['growth_curve'], ('reg_land',))
    # Verify the arithmetic calibration environment against the 2000-2019
    # common-environment mean, retaining the original member and region order.
    with xr.open_dataset(data_root / SOURCES['common_environment']) as common:
        expected = common[list(ENVIRONMENT)].sel(year=slice(2000, 2019)).mean('year', skipna=False).load()
    exact_coordinates(primary, expected)
    calibration_errors = {}
    for name in ENVIRONMENT:
        a, b = xr.align(expected[name], data['calibration'][name], join='exact')
        error = float(abs(a-b).max())
        if not np.isfinite(error) or error > 1e-12:
            raise ValueError('Calibration environment differs from its source: ' + name)
        calibration_errors[name] = error
        value = 1. if name == 'r_Pl' else 0.
        if not bool((data['pi'][name] == value).all()):
            raise ValueError('Parameter PI environment differs: ' + name)
    lands = {}
    for name in ('history', 'vl', 'm', 'h'):
        path = data_root / f'data/processed/regional_forest_age_forcing_v1/{name}/forcing.nc'
        lands[name] = xr.load_dataset(path)
        exact_coordinates(primary, lands[name], ('reg_land',))
        for field in SEMANTICS:
            xr.testing.assert_equal(lands['history'][field], lands[name][field])
    prepared, static = prepare_primary(primary, data['historical_atmosphere'], lands['history'])
    outputs = {'primary_parameters.nc': prepared,
               'land_metadata.nc': lands['history'][list(SEMANTICS) + ['Aland_initial', 'class_source_declared_available']],
               'prehistory_forcing.nc': regional_forcing(lands['history'], 850, 1750),
               'historical_forcing.nc': regional_forcing(lands['history'], 1750, 2022, data['historical_atmosphere']),
               'growth_curve.nc': data['growth_curve'], 'wood_calibration.nc': data['wood_calibration']}
    for name, role in [('calibration', 'Fixed modern 2000-2019 arithmetic common-environment mean'), ('pi', 'Parameter PI: D_CO2=0, D_Tl=0, r_Pl=1')]:
        env = data[name].copy(deep=True)
        env.attrs = {'role': role, 'source': SOURCES[name], 'validation': 'Values checked against declared source; not model validation'}
        outputs[name + '_environment.nc'] = env
    scenarios = tuple(str(x) for x in data['future_atmosphere'].scen.values)
    if scenarios != ('scen7-VL', 'scen7-M', 'scen7-H'):
        raise ValueError('Future CMIP7 scenario order differs: ' + str(scenarios))
    for scenario, name in zip(scenarios, ('vl', 'm', 'h')):
        outputs[f'future_{name}_forcing.nc'] = regional_forcing(lands[name], 2022, 2099,
                                                               data['future_atmosphere'].sel(scen=scenario, drop=True))
    output.mkdir(parents=True)
    reports = {}
    for name, dataset in outputs.items():
        path = output / name
        dataset.to_netcdf(path, engine='netcdf4')
        with xr.open_dataset(path) as reopened:
            xr.testing.assert_identical(dataset, reopened.load())
        reports[name] = fingerprint(path)
    report = {'status': 'prepared_inputs_verified', 'model_integration_verified': False,
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'preparer': fingerprint(__file__), 'ordered_source_ids_sha256': digest,
              'source_config': primary.source_config.values.tolist(),
              'source_artifacts': {name: fingerprint(data_root / path) for name, path in SOURCES.items() if name != 'common_environment'},
              'common_environment_identity_from_existing_record': data['calibration'].attrs['common_environment_sha256'],
              'calibration_source_comparison_max_abs_error': calibration_errors,
              'inherited_calibration_pipeline_status': data['calibration'].attrs.get('status'),
              'static_forcing_fields_considered': static,
              'legacy_land_static_removed': list(LEGACY_LAND_STATIC), 'legacy_land_dynamic_removed': list(LEGACY_LAND_DYNAMIC),
              'initial_global_area_Mha': float(prepared.Aland_pi.sum()),
              'initial_primary_forest_Mha': float(prepared.Aland_pi.sel(bio_land='Forest').sum()),
              'initial_cohort_forest_Mha': 0., 'data_lineage': {n: dict(d.attrs) for n, d in lands.items()},
              'outputs': reports,
              'limitations': ['Input preparation does not accept ecology or full model integration.',
                              'Age-production response and preference switches remain explicit experiment choices.',
                              'Future atmospheric inputs preserve source coverage; paired emissions experiments require a declared emissions protocol.',
                              'Initialize zero carbon, dynamically spin up, then install consistent PI stock/fire baselines.']}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.data_root, args.output)
    print(json.dumps({k: report[k] for k in ('status', 'ordered_source_ids_sha256', 'initial_primary_forest_Mha')}))


if __name__ == '__main__':
    main()

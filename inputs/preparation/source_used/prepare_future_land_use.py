"""Reduce future LUH3 actual events, independently of any climate background."""
import argparse
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_land_use import CLASSES, HARVEST, check, integrate, region_sum, save
from audit_future_land_compatibility import SOURCE_URL, metrics, schema, regional_sum as support_sum


def reduce_stored(value, support, area, weights, *, mass=False, unavailable=False):
    """Strict storage reduction; unavailable zeros require a separate semantic mask."""
    finite = np.isfinite(value)
    check(unavailable or finite[support].all(), 'Missing storage on accepted land support')
    check(not np.any((~support) & finite & (value != 0)), 'Nonzero outside historical support')
    valid = value[support & finite]
    if unavailable:
        check(np.all(valid == 0), 'Unavailable plantation storage changed; review representation')
    masked = np.where(support & finite, value, 0.)
    scale = 1e-12 if mass else area
    reduced = region_sum(masked * scale, weights)
    correction = region_sum(np.where(masked < 0, -masked, 0.) * scale, weights) if np.any(valid < 0) else np.zeros_like(reduced)
    if unavailable and np.any(support & ~finite):
        missing_area = support_sum(np.where(support & ~finite, area, 0.), weights)
        reduced[missing_area > 0] = np.nan
    return reduced, correction, int((valid < 0).sum()), float(valid.min()) if valid.size else np.nan


def demand_and_incoming(transitions, harvest, pairs):
    demand = np.zeros((transitions.shape[0], len(CLASSES)))
    incoming = np.zeros_like(demand)
    for index, (source, target) in enumerate(pairs):
        demand[:, source] += transitions[:, index]
        incoming[:, target] += transitions[:, index]
    sources = ('primf', 'primn', 'secdf', 'secdf', 'secdn', 'pltns')
    targets = ('secdf', 'secdn', 'secdf', 'secdf', 'secdn', 'pltns')
    for index, (source, target) in enumerate(zip(sources, targets)):
        demand[:, CLASSES.index(source)] += harvest[:, index]
        incoming[:, CLASSES.index(target)] += harvest[:, index]
    return demand, incoming


def supply_screen(state, transitions, harvest, pairs, tolerance):
    """Every atomic substep consumes beginning pools; returning harvest area is incoming."""
    demand, incoming = demand_and_incoming(transitions, harvest, pairs)
    net = incoming - demand
    deficits = []
    for count in (1, 2, 4, 8):
        # A linear state path reaches its minimum for each pool at one endpoint.
        minimum_start = np.minimum(state, state + (count - 1) / count * net)
        deficits.append(float(np.maximum(demand / count - minimum_start, 0.).max()))
    feasible = next((n for n, d in zip((1, 2, 4, 8), deficits) if d <= tolerance), 0)
    return demand, incoming, deficits, feasible


def accept_metrics(part, limits):
    """Numerical acceptance is explicit and independent of source caveats."""
    gates = {
        'finite_nonnegative_cumulative_states': part['minimum_state_Mha'] >= 0,
        'regional_land_closure': part['max_land_closure_Mha'] <= limits['land_closure_Mha'],
        'historical_overlap': max(v['max_abs_delta'] for v in part['overlap'].values()) <= limits['overlap_absolute'],
        'small_conversion_correction': part['conversion_correction_total_Mha'] <= limits['conversion_correction_total_Mha'],
        'small_negative_conversion_fraction': part['minimum_conversion_fraction'] >= -limits['negative_conversion_fraction'],
        'small_published_state_difference': part['published_comparison']['max_abs_delta'] <= limits['published_difference_Mha'],
        'annual_supply_with_fixed_substeps': all(n > 0 for n in part['minimum_feasible_substeps']),
    }
    return gates


def prepare(root, spec, scenario, out, report):
    receipt = json.loads((root / spec['acquisition_receipt']).read_text())
    selected = [item for item in receipt['completed_or_failed'] if Path(item['path']).parent.name == scenario]
    check(len(selected) == 2, 'Need exactly two completed scenario files')
    paths = {}
    for item in selected:
        check(item['byte_validation'] == item['container_open'] == 'passed', 'Invalid acquisition receipt')
        check((root / item['path']).stat().st_size == item['bytes'], 'Raw file size changed')
        paths['states' if Path(item['path']).name.startswith('multiple-states') else 'transitions'] = root / item['path']
    report['source_receipts'] = selected
    with ExitStack() as stack:
        def opened(path):
            return stack.enter_context(xr.open_dataset(path, decode_times=False))
        hist = opened(root / spec['historical_area'])
        historical_root = root / spec['historical_raw_directory']
        hs = opened(next(historical_root.glob('multiple-states*.nc')))
        static = opened(next(historical_root.glob('multiple-static*.nc')))
        rw = opened(root / spec['region_weights'])
        fs, ft = opened(paths['states']), opened(paths['transitions'])
        regions = hist.reg_land.values.tolist()
        transfers = hist.transition.values.tolist()
        check(len(transfers) == 130 and set(transfers) == set(n for n in ft if '_to_' in n), 'Conversion schema changed')
        check(hist.luh_class.values.tolist() == list(CLASSES), 'Historical class ordering differs')
        check(hist.harvest_source.values.tolist() == list(HARVEST), 'Historical harvest ordering differs')
        check(regions == rw.reg_land.values.tolist(), 'Region labels differ')
        for coordinate in ('lat', 'lon'):
            check(np.array_equal(rw[coordinate], hs[coordinate]), 'Weights grid differs')
            check(np.array_equal(static[coordinate], hs[coordinate]), 'Static grid differs')
        check(static.carea.attrs['units'] == 'km2', 'Static area units changed')
        weights = np.nan_to_num(rw.region_fraction.values, nan=0.)
        check(weights.shape == (180, 360, 11), 'Unexpected region weights')
        area = np.asarray(static.carea, dtype=np.float64) / 10000.
        check(np.isfinite(area).all() and np.all(area > 0), 'Invalid whole-cell area')
        support = np.isfinite(hs.primf.isel(time=0).values)
        land = np.where(support, 1 - np.asarray(static.icwtr, dtype=np.float64), 0.)
        static_region = region_sum(land * area, weights)
        event_names = transfers + [h + '_harv' for h in HARVEST] + [h + '_bioh' for h in HARVEST]
        report['schema'] = {
            'states': schema(fs, hs, 2022, 2100, CLASSES, dict.fromkeys(CLASSES, '1')),
            'transitions': schema(ft, hs, 2022, 2099, event_names, {n: 'kg' if n.endswith('_bioh') else '1' for n in event_names})}
        for ds in (fs, ft):
            check(ds.attrs['source_id'] == spec['source_ids'][scenario], 'Wrong source identity')
            check('plantation' in ds.attrs.get('comment', '').lower(), 'Missing source plantation caveat')
        check(fs.attrs.get('iam_scenario') == ft.attrs.get('iam_scenario'), 'IAM scenarios differ')
        pairs = [(CLASSES.index(n.split('_to_')[0]), CLASSES.index(n.split('_to_')[1])) for n in transfers]
        years, intervals = np.arange(2022, 2101), np.arange(2022, 2100)
        state = np.zeros((79, 11, 13))
        published = np.zeros_like(state)
        transition = np.zeros((78, 11, 130))
        raw_transition = np.zeros_like(transition)
        correction = np.zeros_like(transition)
        harvest = np.zeros((78, 11, 6))
        harvest_stored = np.zeros_like(harvest)
        plantation_state_finite = np.zeros(79, dtype=np.int64)
        plantation_event_finite = np.zeros((78, 24), dtype=np.int64)
        carbon = np.zeros_like(harvest)
        negative_carbon_mass = np.zeros_like(harvest)
        state_count = np.zeros((79, 13), dtype=np.int64)
        state_min = np.zeros((79, 13))
        transition_count = np.zeros((78, 130), dtype=np.int64)
        transition_min = np.zeros((78, 130))
        carbon_count = np.zeros((78, 6), dtype=np.int64)
        carbon_min = np.zeros((78, 6))
        partition = np.zeros(79)
        supply = np.zeros_like(state[:-1])
        incoming = np.zeros_like(supply)
        deficits = np.zeros((78, 4))
        feasible = np.zeros(78, dtype=np.int64)
        state[0] = hist.state_area.sel(state_year=2022).values
        check(np.isfinite(state[0]).all() and np.all(state[0] >= 0), 'Invalid accepted historical anchor')
        check(np.all(state[0, :, -1] == 0), 'Dormant plantation representation requires zero historical anchor')
        started = time.monotonic()
        for i, year in enumerate(years):
            total = np.zeros_like(area)
            for j, name in enumerate(CLASSES):
                value = np.asarray(fs[name].isel(time=i), dtype=np.float64)
                published[i, :, j], _, state_count[i, j], state_min[i, j] = reduce_stored(value, support, area, weights, unavailable=name == 'pltns')
                if name == 'pltns':
                    plantation_state_finite[i] = np.isfinite(value[support]).sum()
                if name != 'pltns':
                    total += np.where(support, value, 0.)
            partition[i] = np.abs(total[support] - land[support]).max()
            if year == 2100:
                break
            for j, name in enumerate(transfers):
                value = np.asarray(ft[name].isel(time=i), dtype=np.float64)
                raw_transition[i, :, j], correction[i, :, j], transition_count[i, j], transition_min[i, j] = reduce_stored(value, support, area, weights, unavailable='pltns' in name)
                # Match historical accumulation: correct cells before regional summation.
                transition[i, :, j] = 0. if 'pltns' in name else region_sum(np.where(support, np.maximum(value, 0.), 0.) * area, weights)
                if 'pltns' in name:
                    plantation_event_finite[i, [n for n in transfers if 'pltns' in n].index(name)] = np.isfinite(value[support]).sum()
            for j, name in enumerate(HARVEST):
                value = np.asarray(ft[name + '_harv'].isel(time=i), dtype=np.float64)
                harvest_stored[i, :, j], _, count, _ = reduce_stored(value, support, area, weights, unavailable=name == 'pltns')
                check(count == 0, 'Negative harvest area requires a new data decision')
                harvest[i, :, j] = 0. if name == 'pltns' else harvest_stored[i, :, j]
                if name == 'pltns':
                    plantation_event_finite[i, 22] = np.isfinite(value[support]).sum()
                value = np.asarray(ft[name + '_bioh'].isel(time=i), dtype=np.float64)
                if name == 'pltns':
                    plantation_event_finite[i, 23] = np.isfinite(value[support]).sum()
                carbon[i, :, j], negative_carbon_mass[i, :, j], carbon_count[i, j], carbon_min[i, j] = reduce_stored(value, support, area, weights, mass=True, unavailable=name == 'pltns')
            state[i + 1], _ = integrate(state[i], transition[i], harvest[i], pairs)
            supply[i], incoming[i], deficits[i], feasible[i] = supply_screen(state[i], transition[i], harvest[i], pairs, spec['acceptance']['supply_tolerance_Mha'])
            report.update(last_completed_interval=int(year), elapsed_seconds=time.monotonic() - started,
                          minimum_state_Mha=float(state[:i+2].min()), conversion_correction_total_Mha=float(correction[:i+1].sum()))
            save(out / 'report.json', report)
            print(scenario, int(year), 'elapsed', round(report['elapsed_seconds'], 1), 'minimum area', report['minimum_state_Mha'], flush=True)
        available_class = np.array([c != 'pltns' for c in CLASSES])
        available_transition = np.array(['pltns' not in n for n in transfers])
        available_harvest = np.array([h != 'pltns' for h in HARVEST])
        published_available = np.where(available_class, published, np.nan)
        report['overlap'] = {
            'published_state': metrics(published_available[:3], hist.published_state_area.sel(state_year=years[:3]).values, list(CLASSES), regions, years[:3]),
            'corrected_cumulative_state': metrics(state[:3], hist.state_area.sel(state_year=years[:3]).values, list(CLASSES), regions, years[:3]),
            'raw_transition': metrics(np.where(available_transition, raw_transition[:2], np.nan), (hist.transition_area - hist.transition_negative_correction).sel(interval=intervals[:2]).values, transfers, regions, intervals[:2]),
            'harvest_area': metrics(np.where(available_harvest, harvest[:2], np.nan), hist.harvest_area.sel(interval=intervals[:2]).values, list(HARVEST), regions, intervals[:2]),
            'signed_harvest_carbon': metrics(np.where(available_harvest, carbon[:2], np.nan), hist.harvest_carbon_raw.sel(interval=intervals[:2]).values, list(HARVEST), regions, intervals[:2])}
        report.update(
            regions=regions, support_cells=int(support.sum()), static_land_Mha=static_region.tolist(),
            published_comparison=metrics(np.where(available_class, state, np.nan), published_available, list(CLASSES), regions, years),
            minimum_state_Mha=float(state.min()), max_land_closure_Mha=float(np.abs(state.sum(axis=2) - static_region).max()),
            conversion_correction_total_Mha=float(correction.sum()), conversion_negative_cell_occurrences=int(transition_count.sum()),
            minimum_conversion_fraction=float(np.nanmin(transition_min)), max_annual_conversion_correction_Mha=float(correction.sum(axis=(1, 2)).max()),
            signed_bioh_negative_cell_occurrences=int(carbon_count.sum()), signed_bioh_minimum_kg=float(np.nanmin(carbon_min)),
            negative_bioh_absolute_PgC=float(negative_carbon_mass.sum()), bioh_correction_applied=False,
            max_observed_12_class_partition_fraction=float(partition.max()), minimum_feasible_substeps=feasible.tolist(),
            maximum_source_supply_deficit_Mha_by_substeps=dict(zip(('1', '2', '4', '8'), deficits.max(axis=0).tolist())),
            max_demand_minus_beginning_and_incoming_Mha=float(np.maximum(supply - state[:-1] - incoming, 0.).max()),
            plantation_representation='Finite processed zero for dormant unrepresented class; source availability false. Raw zero storage (2022/2023), later CF missing storage, and semantic NaN views retained. No allocation inferred.',
            H_warning='Official early-H allocation warning remains unresolved; nonforest harvest is not a forest reset and no correction is attempted.',
            first_experiment_state_end=2099, first_experiment_interval_end=2098,
            archive_only_interval='2099 to 2100', overlap_policy='Historical 2022 accumulated anchor, future intervals 2022 onward exactly once; do not append historical 2022/2023 events again.')
        report['plantation_state_finite_supported_cells_by_year'] = plantation_state_finite.tolist()
        report['plantation_event_finite_supported_cells_by_year_and_field'] = plantation_event_finite.tolist()
        if scenario == 'h':
            r, i = regions.index('PAC_OECD'), 2025 - 2022
            j = [HARVEST.index('primn'), HARVEST.index('secnf')]
            report['H_2025_PAC_OECD_nonforest_signal'] = {'area_Mha': float(harvest[i, r, j].sum()), 'signed_carbon_PgC': float(carbon[i, r, j].sum()), 'attributed_error': False}
        report['acceptance_gates'] = accept_metrics(report, spec['acceptance'])
        report['status'] = 'accepted_with_source_caveats' if all(report['acceptance_gates'].values()) else 'complete_requires_review'
        ds = xr.Dataset(coords={'state_year': years, 'interval': intervals, 'reg_land': regions,
            'luh_class': list(CLASSES), 'transition': transfers, 'harvest_source': list(HARVEST), 'bounds': [0, 1], 'substeps': [1, 2, 4, 8]})
        variables = [
            ('state_area', state, ('state_year', 'reg_land', 'luh_class'), 'Mha'),
            ('published_state_area_stored', published, ('state_year', 'reg_land', 'luh_class'), 'Mha'),
            ('published_state_area', published_available, ('state_year', 'reg_land', 'luh_class'), 'Mha'),
            ('state_minus_published', np.where(available_class, state - published, np.nan), ('state_year', 'reg_land', 'luh_class'), 'Mha'),
            ('transition_area', transition, ('interval', 'reg_land', 'transition'), 'Mha'),
            ('transition_area_raw_stored', raw_transition, ('interval', 'reg_land', 'transition'), 'Mha'),
            ('transition_area_raw_available', np.where(available_transition, raw_transition, np.nan), ('interval', 'reg_land', 'transition'), 'Mha'),
            ('transition_negative_correction', correction, ('interval', 'reg_land', 'transition'), 'Mha'),
            ('harvest_area_stored', harvest_stored, ('interval', 'reg_land', 'harvest_source'), 'Mha'),
            ('harvest_area', harvest, ('interval', 'reg_land', 'harvest_source'), 'Mha'),
            ('harvest_area_available', np.where(available_harvest, harvest, np.nan), ('interval', 'reg_land', 'harvest_source'), 'Mha'),
            ('harvest_carbon_raw', carbon, ('interval', 'reg_land', 'harvest_source'), 'Pg C'),
            ('harvest_carbon_raw_available', np.where(available_harvest, carbon, np.nan), ('interval', 'reg_land', 'harvest_source'), 'Pg C'),
            ('harvest_carbon_negative_absolute', negative_carbon_mass, ('interval', 'reg_land', 'harvest_source'), 'Pg C'),
            ('raw_state_negative_count', state_count, ('state_year', 'luh_class'), '1'),
            ('raw_state_min_fraction', state_min, ('state_year', 'luh_class'), '1'),
            ('raw_transition_negative_count', transition_count, ('interval', 'transition'), '1'),
            ('raw_transition_min_fraction', transition_min, ('interval', 'transition'), '1'),
            ('raw_bioh_negative_count', carbon_count, ('interval', 'harvest_source'), '1'),
            ('raw_bioh_min_kgC', carbon_min, ('interval', 'harvest_source'), 'kg C'),
            ('raw_observed_partition_max_abs_fraction', partition, ('state_year',), '1'),
            ('source_demand_area', supply, ('interval', 'reg_land', 'luh_class'), 'Mha'),
            ('source_incoming_area', incoming, ('interval', 'reg_land', 'luh_class'), 'Mha'),
            ('maximum_source_deficit', deficits, ('interval', 'substeps'), 'Mha')]
        for name, value, dims, units in variables:
            ds[name] = (dims, value, {'units': units})
        ds['plantation_state_stored_finite_count'] = ('state_year', plantation_state_finite)
        ds['plantation_event_stored_finite_count'] = (('interval', 'plantation_event'), plantation_event_finite)
        ds = ds.assign_coords(plantation_event=[n for n in transfers if 'pltns' in n] + ['pltns_harv', 'pltns_bioh'])
        ds['minimum_feasible_substeps'] = ('interval', feasible)
        for label, mask, dim in [('state', available_class, 'luh_class'), ('transition', available_transition, 'transition'), ('harvest', available_harvest, 'harvest_source')]:
            ds[label + '_source_declared_available'] = (dim, mask.astype('int8'))
        ds['harvest_carbon_source_declared_available'] = ('harvest_source', available_harvest.astype('int8'))
        ds['interval_start'] = ('interval', intervals)
        ds['interval_end'] = ('interval', intervals + 1)
        ds['first_experiment_interval'] = ('interval', (intervals <= 2098).astype('int8'))
        ds['first_experiment_state'] = ('state_year', (years <= 2099).astype('int8'))
        for key, source, dim in [('state', fs, 'state_year'), ('event', ft, 'interval')]:
            attrs = {k: v for k, v in source.time.attrs.items() if k != 'bounds'}
            ds[key + '_source_time'] = (dim, source.time.values, attrs)
            ds[key + '_source_time_bounds'] = ((dim, 'bounds'), source[source.time.attrs['bounds']].values, attrs)
        ds['transition_from'] = ('transition', [n.split('_to_')[0] for n in transfers])
        ds['transition_to'] = ('transition', [n.split('_to_')[1] for n in transfers])
        ds.attrs.update(scenario=scenario.upper(), iam_scenario=fs.attrs.get('iam_scenario', ''),
            source_id=fs.attrs['source_id'], source_version=fs.attrs['source_version'], source_doi=fs.attrs.get('doi', ''),
            historical_anchor=spec['historical_area'] + ':state_area(2022)', region_weights=spec['region_weights'],
            state_policy='Accumulate corrected original gross conversions and primary harvest from accepted historical 2022. No annual reset or fabricated balancing events.',
            plantation_policy=report['plantation_representation'], carbon_policy='Signed published bioh diagnostic only; no clipping, model carbon demand, or inferred reset.',
            event_policy='Historical-compatible cellwise correction of negative conversions, additive correction separately retained.',
            time_semantics='Preserved source annual coordinates and bounds; time: mean does not establish instantaneous stock sampling.',
            H_warning=report['H_warning'], data_status=report['status'], first_experiment_end=2099)
        ds.to_netcdf(out / 'land_use_absolute.nc', encoding={n: {'zlib': True, 'complevel': 2} for n in ds if ds[n].dtype.kind in 'fiu'})
        report['output'] = str((out / 'land_use_absolute.nc').relative_to(root))
        report['output_sha256'] = hashlib.sha256((out / 'land_use_absolute.nc').read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path('config/future_land_use.json'))
    parser.add_argument('--scenario', required=True, choices=('vl', 'm', 'h', 'hl', 'ln', 'l', 'ml'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads((root / args.config).read_text())
    out = root / (args.out or Path(spec['output_directory']) / args.scenario)
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'scenario': args.scenario.upper(), 'policy': spec,
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'official_documentation': SOURCE_URL, 'primary_source_verification': 'Land-use variable interpretation follows the official LUH3 documentation linked in official_documentation.',
        'raw_rehashed': False, 'source_files_modified': False, 'model_executed': False, 'official_background_land_read': False}
    try:
        prepare(root, spec, args.scenario, out, report)
    except BaseException as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save(out / 'report.json', report)


if __name__ == '__main__':
    main()

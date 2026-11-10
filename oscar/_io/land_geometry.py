"""Small geometry manifests and one-year NetCDF plans; no LUH interpretation."""
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import xarray as xr
from .._core._model._modules.land_geometry import (
    LandClass, EventSpec, SourceKey, AreaTable, SelectedArea, RegionalPlan, GeometryState,
)


def load_semantics(data):
    if data.get('schema') != 'prepared-land-events-v1':
        raise ValueError('Unsupported prepared semantics schema')
    classes = tuple(LandClass(**x) for x in data['classes'])
    fields = EventSpec.__dataclass_fields__
    events = tuple(EventSpec(**{k: v for k, v in x.items() if k in fields}) for x in data['events'])
    if len({c.class_id for c in classes}) != len(classes) or len({e.event_id for e in events}) != len(events):
        raise ValueError('Duplicate class/event label')
    return classes, events


def write_plan(path, plan):
    """Write one pure plan; metadata definitions live in the small manifest."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    data = xr.Dataset(coords={'region': list(plan.region_ids), 'event': [e.event_id for e in plan.events]})
    for prefix, table in [('opening', plan.opening), ('surviving', plan.surviving), ('closing', plan.closing)]:
        dim = prefix + '_key'
        data[prefix + '_class'] = (dim, [k.current_class for k in table.keys])
        data[prefix + '_birth'] = (dim, [np.nan if k.birth_time is None else k.birth_time for k in table.keys])
        data[prefix + '_origin'] = (dim, ['' if k.entry_origin is None else k.entry_origin for k in table.keys])
        data[prefix + '_area'] = (('region', dim), table.area, {'units': 'Mha'})
    for field in ('region_index', 'event_index', 'source_index', 'area'):
        data['selected_' + field] = ('flow', getattr(plan.selected, field))
    for field in ('requested_area', 'executed_area'):
        data[field] = (('region', 'event'), getattr(plan, field), {'units': 'Mha'})
    data['initial_land_area'] = ('region', plan.initial_land_area, {'units': 'Mha'})
    data.attrs = dict(schema='land-geometry-plan-v4', **{name: getattr(plan, name) for name in
        ('strategy', 'interval_start', 'interval_end', 'event_time', 'environment_year', 'next_year',
         'adjusted_native_sources', 'removed_native_requests')})
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)


def read_plan(path, classes, events, region_ids):
    """Read and validate one complete regional plan, without a calendar scan."""
    with xr.open_dataset(path) as d:
        if d.attrs.get('schema') != 'land-geometry-plan-v4':
            raise ValueError('Unsupported annual geometry schema')
        if tuple(d.region.values.tolist()) != tuple(region_ids):
            raise ValueError('Annual geometry regions differ from manifest')
        lookup = {e.event_id: e for e in events}
        labels = d.event.values.tolist()
        if len(labels) != len(lookup) or set(labels) != set(lookup):
            raise ValueError('Annual geometry events differ from manifest')
        def table(prefix):
            required = {prefix + suffix for suffix in ('_class', '_birth', '_origin', '_area')}
            if not required.issubset(d.variables):
                raise ValueError('Annual geometry missing required ' + prefix + ' table')
            keys = tuple(SourceKey(str(c), None if np.isnan(b) else float(b), o or None)
                for c, b, o in zip(d[prefix+'_class'].values, d[prefix+'_birth'].values,
                                  d[prefix+'_origin'].values.tolist()))
            return AreaTable(keys, d[prefix+'_area'].values)
        selected = SelectedArea(*(d['selected_'+name].values for name in
            ('region_index', 'event_index', 'source_index', 'area')))
        plan = RegionalPlan(tuple(region_ids), tuple(lookup[e] for e in labels), str(d.strategy),
            *(int(d.attrs[k]) for k in ('interval_start', 'interval_end')),
            float(d.event_time), int(d.environment_year), int(d.next_year),
            table('opening'), selected, table('closing'), d.requested_area.values, d.executed_area.values,
            table('surviving'), d.initial_land_area.values,
            int(d.adjusted_native_sources), int(d.removed_native_requests))
    plan.validate_classes(classes)
    return plan


class GeometryInput:
    """Read one annual plan at a time; immutable identity comes from the manifest.

    No scan/hash of all annual files is performed. A manifest declares its own
    source, region-weight and parent-history identities. Input corruption is
    checked against the complete state when each plan is consumed.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.metadata = json.loads(self.path.read_text())
        m = self.metadata
        if m.get('schema') != 'land-geometry-input-v4':
            raise ValueError('Geometry requires a complete-key manifest; legacy input is unsupported')
        for name in ('identity', 'strategy', 'source_identity', 'region_weights_identity'):
            if not isinstance(m.get(name), str) or not m[name]:
                raise ValueError('Missing geometry identity: ' + name)
        self.classes, self.events = load_semantics(m['semantics'])
        self.region_ids = tuple(m['region_ids'])
        self.first_year, self.end_year = m['first_year'], m['end_year']
        if (type(self.first_year) is not int or type(self.end_year) is not int
                or self.end_year <= self.first_year or len(set(self.region_ids)) != len(self.region_ids)
                or not self.region_ids or m['strategy'] not in ('proportional', 'land_use_specific')):
            raise ValueError('Invalid geometry calendar, strategy or regions')
        if set(m['annual_files']) != {str(y) for y in range(self.first_year, self.end_year)}:
            raise ValueError('Geometry manifest has missing/extra annual labels')
        self._cached_year = None
        self._cached_plan = None

    @property
    def identity(self):
        return self.metadata['identity']

    def plan(self, year, regions=None):
        if type(year) is not int or not self.first_year <= year < self.end_year:
            raise ValueError('Geometry year outside manifest coverage')
        if self._cached_year != year:
            entry = self.metadata['annual_files'][str(year)]
            file = self.path.parent / entry['path']
            if hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Current annual geometry content identity changed')
            plan = read_plan(file, self.classes, self.events, self.region_ids)
            if plan.interval_start != year or plan.strategy != self.metadata['strategy']:
                raise ValueError('Annual geometry calendar/strategy differs from manifest')
            self._cached_year, self._cached_plan = year, plan
        plan = self._cached_plan
        if regions is None or tuple(regions) == plan.region_ids:
            return plan
        regions = tuple(regions)
        if len(set(regions)) != len(regions) or any(r not in self.region_ids for r in regions):
            raise ValueError('Unknown/duplicate requested geometry region')
        indices = [self.region_ids.index(r) for r in regions]
        mapping = {old: new for new, old in enumerate(indices)}
        f = plan.selected
        mask = np.isin(f.region_index, indices)
        selected = SelectedArea(np.array([mapping[int(r)] for r in f.region_index[mask]], dtype=int),
                                f.event_index[mask], f.source_index[mask], f.area[mask])
        return replace(plan, region_ids=regions, opening=AreaTable(plan.opening.keys, plan.opening.area[indices]),
            closing=AreaTable(plan.closing.keys, plan.closing.area[indices]), selected=selected,
            surviving=AreaTable(plan.surviving.keys, plan.surviving.area[indices]),
            requested_area=plan.requested_area[indices], executed_area=plan.executed_area[indices],
            initial_land_area=plan.initial_land_area[indices])

    def boundary(self, year, regions=None):
        if year == self.end_year:
            return self.plan(year - 1, regions).closing
        return self.plan(year, regions).opening


def write_geometry(path, plans, classes, *, identity, source_identity, region_weights_identity,
                   parent_geometry_identity=None):
    """Small prepared/synthetic writer; production batch orchestration is external."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    annual = {}
    first = None
    for plan in plans:
        plan.validate_classes(classes)
        if first is None:
            first = plan
        if (plan.events != first.events or plan.region_ids != first.region_ids or plan.strategy != first.strategy
                or plan.interval_start != first.interval_start + len(annual)):
            raise ValueError('Plans must have consecutive years and common definitions')
        name = f'{path.stem}_{plan.interval_start}.nc'
        write_plan(path.parent / name, plan)
        annual[str(plan.interval_start)] = dict(path=name, sha256=hashlib.sha256((path.parent/name).read_bytes()).hexdigest())
    if first is None:
        raise ValueError('Geometry cannot be empty')
    metadata = dict(schema='land-geometry-input-v4', identity=identity, source_identity=source_identity,
        region_weights_identity=region_weights_identity, parent_geometry_identity=parent_geometry_identity,
        strategy=first.strategy, region_ids=list(first.region_ids), first_year=first.interval_start,
        end_year=first.interval_start+len(annual), annual_files=annual,
        semantics=dict(schema='prepared-land-events-v1', classes=[vars(c) for c in classes],
                       events=[vars(e) for e in first.events]))
    path.write_text(json.dumps(metadata, indent=2)+'\n')
    return GeometryInput(path)


def save_native_state(path, state, *, metadata):
    """Atomically store native arrays, without expanding area into Python lists."""
    if not isinstance(state, GeometryState):
        raise TypeError('Native checkpoint requires GeometryState')
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = xr.Dataset({
        'current_class': ('key', [k.current_class for k in state.keys]),
        'birth_time': ('key', [np.nan if k.birth_time is None else k.birth_time for k in state.keys]),
        'entry_origin': ('key', [k.entry_origin or '' for k in state.keys]),
        'area': (('cell', 'key'), state.area, {'units': 'Mha'}),
        'initial_land_area': ('cell', state.initial_land_area, {'units': 'Mha'}),
        'numerical_area_reduction': ('cell', state.numerical_area_reduction, {'units': 'Mha'})},
        coords={'cell': list(state.cell_ids)},
        attrs=dict(schema='land-geometry-v3', next_year=state.next_year, strategy=state.strategy,
            classes=json.dumps([vars(c) for c in state.classes]),
            metadata=json.dumps(metadata, sort_keys=True)))
    data.to_netcdf(temporary, engine='h5netcdf')
    restored, _ = load_native_state(temporary, expected_metadata=metadata)
    if (restored.keys != state.keys or restored.cell_ids != state.cell_ids
            or restored.next_year != state.next_year or restored.strategy != state.strategy
            or not np.array_equal(restored.area, state.area)
            or not np.array_equal(restored.initial_land_area, state.initial_land_area)
            or not np.array_equal(restored.numerical_area_reduction, state.numerical_area_reduction)):
        raise ValueError('Native checkpoint array readback differs')
    os.link(temporary, path)
    temporary.unlink()


def load_native_state(path, *, expected_metadata=None):
    """Read complete native keys, cell order, area and integer-year position."""
    with xr.open_dataset(path) as data:
        if data.attrs.get('schema') != 'land-geometry-v3':
            raise ValueError('Unsupported native checkpoint schema')
        metadata = json.loads(data.attrs['metadata'])
        if expected_metadata is not None and metadata != expected_metadata:
            raise ValueError('Native checkpoint input/partition identity differs')
        if data.area.dtype != np.float64 or data.area.attrs.get('units') != 'Mha':
            raise ValueError('Native checkpoint must retain float64 Mha arrays')
        if (not isinstance(data.attrs['next_year'], (int, np.integer)) or data.cell.dtype.kind not in 'iu'):
            raise ValueError('Native checkpoint year and cell IDs must be integers')
        for field in ('initial_land_area', 'numerical_area_reduction'):
            if data[field].dtype != np.float64 or data[field].attrs.get('units') != 'Mha':
                raise ValueError('Native numerical history must retain float64 Mha')
        keys = tuple(SourceKey(str(c), None if np.isnan(b) else float(b), o or None)
            for c,b,o in zip(data.current_class.values, data.birth_time.values, data.entry_origin.values.tolist()))
        state = GeometryState(tuple(LandClass(**c) for c in json.loads(data.attrs['classes'])),
            tuple(int(c) for c in data.cell.values), keys, data.area.transpose('cell','key').values,
            int(data.attrs['next_year']), str(data.attrs['strategy']),
            data.initial_land_area.values, data.numerical_area_reduction.values)
    return state, metadata

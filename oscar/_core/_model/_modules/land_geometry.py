"""Native-cell area and forest-age geometry, independent of carbon and LUH I/O.

All areas are float64 Mha. Cells evolve once; region weights are applied only
when recording the annual plan. This prototype is not wired to production.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import fsum

import numpy as np


ENTRY_ORIGINS = ("cleared", "primn", "secdn", "range", "c3ann", "c4ann",
                 "c3per", "c4per", "c3nfx", "pastr", "urban")
STRATEGIES = ("proportional", "land_use_specific")
SCHEMA = "land-geometry-v3"
NUMERICAL_RTOL = 1e-10  # Engineering tolerance relative to fixed initial cell land.


def sum_rows(values):
    """Sum each row along a contiguous fast axis using NumPy pairwise summation.

    Advanced column indexing can produce Fortran-order arrays. Summing their
    long key axis directly uses a less accurate accumulation order. This helper
    changes only the reduction layout, not values or acceptance tolerances.
    """
    return np.ascontiguousarray(values).sum(axis=1)


def _roundoff(a, b):
    # Existing event-kernel precision: relative, with no absolute Mha floor.
    return 16 * np.finfo(np.float64).eps * np.maximum(np.abs(a), np.abs(b))


def _array(value, shape, name):
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != shape or not np.isfinite(out).all() or np.any(out < 0):
        raise ValueError(f"{name} must have shape {shape}, finite nonnegative values")
    out.setflags(write=False)
    return out


def _ids(values, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    return values


@dataclass(frozen=True)
class LandClass:
    class_id: str
    biome: str
    role: str

    def __post_init__(self):
        if not self.class_id or not self.biome or self.role not in (
                "forest_unaged", "forest_cohort", "nonforest"):
            raise ValueError("invalid land class role or identity")


@dataclass(frozen=True)
class EventSpec:
    """Prepared semantics, with the same fields as the existing event kernel."""
    event_id: str
    source_class: str
    target_class: str
    kind: str
    vegetation_action: str
    age_action: str
    selection: str
    retained_split: str = "none"

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (
                self.event_id, self.source_class, self.target_class)):
            raise ValueError("event identities must be nonempty strings")
        for value, allowed in (
            (self.kind, ("conversion", "harvest")),
            (self.vegetation_action, ("clear", "retain", "unresolved")),
            (self.age_action, ("reset", "keep", "none")),
            (self.selection, ("proportional", "oldest", "youngest")),
            (self.retained_split, ("none", "source_p_wood")),
        ):
            if value not in allowed:
                raise ValueError(f"invalid event field: {value}")


def _validate_spec(spec, source, target):
    if spec.vegetation_action == "unresolved":
        raise ValueError(f"unresolved nonzero event: {spec.event_id}")
    if target.role == "forest_unaged":
        raise ValueError("events cannot create unaged primary forest")
    if spec.kind == "conversion" and spec.selection != "proportional":
        raise ValueError("conversions must use proportional selection")
    if source.role != "forest_cohort" and spec.selection != "proportional":
        raise ValueError("age selection requires a cohort source")
    if spec.kind == "harvest" and (spec.vegetation_action != "clear" or source.biome != target.biome):
        raise ValueError("harvest must clear within its source biome")
    if target.role == "forest_cohort":
        if spec.vegetation_action == "clear" and spec.age_action != "reset":
            raise ValueError("forest clearance must reset age")
        if spec.vegetation_action == "retain":
            if source.role == "forest_unaged":
                raise ValueError("retained unaged-to-cohort conversion is unsupported")
            if source.role == "forest_cohort" and spec.age_action != "keep":
                raise ValueError("retained forest relabel must preserve birth time")
            if source.role == "nonforest" and (
                    spec.age_action != "reset" or spec.retained_split != "source_p_wood"):
                raise ValueError("nonforest establishment needs reset and source_p_wood")
    elif spec.age_action != "none":
        raise ValueError("nonforest target cannot have a forest age action")
    if source.role != "nonforest" and spec.retained_split != "none":
        raise ValueError("existing forest W/F must not be resplit")


def secondary_stand_harvest(spec):
    """Wood cut that leaves secondary forest standing.

    secdf to secdf clear-harvest removes carbon only. Area and birth year stay.
    New forest (establishment and primf_harv) is not this case.
    """
    return (spec.kind == "harvest" and spec.vegetation_action == "clear"
            and spec.source_class == spec.target_class == "secdf")


@dataclass(frozen=True)
class SourceKey:
    current_class: str
    birth_time: float | None = None
    entry_origin: str | None = None


def _key_order(key, classes):
    return (classes.index(key.current_class),
            -np.inf if key.birth_time is None else key.birth_time,
            -1 if key.entry_origin is None else ENTRY_ORIGINS.index(key.entry_origin))


@dataclass(frozen=True)
class GeometryState:
    """An integer-year boundary, with a cell axis and unique complete keys.

    Primary and nonforest classes each have one fixed key with no birth/origin.
    Finite forest keys carry both birth and origin, regardless of area size.
    Construction accepts any key order and canonically reorders the area axis.
    """
    classes: tuple
    cell_ids: tuple
    keys: tuple
    area: np.ndarray
    next_year: int
    strategy: str
    initial_land_area: np.ndarray | None = None
    numerical_area_reduction: np.ndarray | None = None

    def __post_init__(self):
        classes = tuple(self.classes)
        names = _ids((c.class_id for c in classes), "class IDs")
        roles = {c.class_id: c.role for c in classes}
        if sum(c.role == "forest_unaged" for c in classes) != 1:
            raise ValueError("exactly one primary forest class is required")
        forest_biomes = {c.biome for c in classes if c.role != "nonforest"}
        if len(forest_biomes) != 1 or any(c.biome in forest_biomes for c in classes if c.role == "nonforest"):
            raise ValueError("forest classes must share a separate forest biome")
        cells = _ids(self.cell_ids, "cell IDs")
        if any(not isinstance(c, (int, np.integer)) or isinstance(c, bool) for c in cells):
            raise ValueError("native cell IDs must be integers")
        if type(self.next_year) is not int:
            raise ValueError("geometry snapshots require an integer year boundary")
        if self.strategy not in STRATEGIES:
            raise ValueError("unknown geometry strategy")
        keys = tuple(self.keys)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate complete source key")
        for key in keys:
            if key.current_class not in roles:
                raise ValueError("unknown source class")
            if roles[key.current_class] == "forest_cohort":
                if (key.birth_time is None or not np.isfinite(key.birth_time)
                        or key.birth_time > self.next_year or key.entry_origin not in ENTRY_ORIGINS):
                    raise ValueError("finite forest needs an existing birth and accepted origin")
            elif key.birth_time is not None or key.entry_origin is not None:
                raise ValueError("primary/nonforest keys cannot have a finite age or entry origin")
        fixed = {SourceKey(c.class_id) for c in classes if c.role != "forest_cohort"}
        if not fixed.issubset(keys):
            raise ValueError("all primary/nonforest fixed keys must be present")
        area = _array(self.area, (len(cells), len(keys)), "area")
        order = sorted(range(len(keys)), key=lambda i: _key_order(keys[i], names))
        # Only genuinely empty finite keys can disappear. No area threshold.
        order = [i for i in order if keys[i] in fixed or np.any(area[:, i] > 0)]
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "cell_ids", tuple(int(c) for c in cells))
        object.__setattr__(self, "keys", tuple(keys[i] for i in order))
        object.__setattr__(self, "area", _array(area[:, order], (len(cells), len(order)), "area"))
        initial = sum_rows(self.area) if self.initial_land_area is None else self.initial_land_area
        reduction = np.zeros(len(cells)) if self.numerical_area_reduction is None else self.numerical_area_reduction
        object.__setattr__(self, "initial_land_area", _array(initial, (len(cells),), "initial land area"))
        object.__setattr__(self, "numerical_area_reduction", _array(reduction, (len(cells),), "cumulative numerical reduction"))
        tolerance = NUMERICAL_RTOL * self.initial_land_area
        if np.any(np.abs(sum_rows(self.area)-self.initial_land_area) > tolerance):
            raise ValueError("Native land drift exceeds the fixed initial-land numerical tolerance")
        if np.any(self.numerical_area_reduction > tolerance):
            raise ValueError("Cumulative numerical payment reduction exceeds the fixed initial-land tolerance")


    @classmethod
    def from_class_area(cls, classes, cell_ids, area, *, next_year, strategy):
        """Initialize the native primary state; do not invent secondary ages."""
        classes, cell_ids = tuple(classes), tuple(cell_ids)
        area = _array(area, (len(cell_ids), len(classes)), "class area")
        fixed = [i for i, c in enumerate(classes) if c.role != "forest_cohort"]
        if np.any(area[:, [i for i, c in enumerate(classes) if c.role == "forest_cohort"]] != 0):
            raise ValueError("initial finite forest area needs an explicit ancestry; cannot infer ages")
        return cls(classes, cell_ids, tuple(SourceKey(classes[i].class_id) for i in fixed),
                   area[:, fixed], next_year, strategy)

    @property
    def class_area(self):
        return np.column_stack([sum_rows(self.area[:, [i for i, k in enumerate(self.keys)
                                if k.current_class == c.class_id]])
                                for c in self.classes])

    def snapshot(self):
        return dict(schema=SCHEMA, classes=[vars(c) for c in self.classes],
                    cell_ids=list(self.cell_ids), keys=[vars(k) for k in self.keys],
                    area=self.area.tolist(), next_year=self.next_year, strategy=self.strategy,
                    initial_land_area=self.initial_land_area.tolist(),
                    numerical_area_reduction=self.numerical_area_reduction.tolist())

    @classmethod
    def from_snapshot(cls, data):
        data = dict(data)
        if data.pop("schema", None) != SCHEMA:
            raise ValueError("unsupported geometry snapshot schema; no legacy inference")
        if "initial_land_area" not in data or "numerical_area_reduction" not in data:
            raise ValueError("Geometry snapshot must retain the fixed land scale and cumulative numerical reductions")
        data["classes"] = tuple(LandClass(**c) for c in data["classes"])
        data["keys"] = tuple(SourceKey(**k) for k in data["keys"])
        return cls(**data)


@dataclass(frozen=True)
class AreaTable:
    keys: tuple
    area: np.ndarray  # region, complete source key

    def __post_init__(self):
        keys = tuple(self.keys)
        if len(set(keys)) != len(keys) or any(not isinstance(k, SourceKey) for k in keys):
            raise ValueError("area table requires unique complete source keys")
        for key in keys:
            if not isinstance(key.current_class, str) or not key.current_class:
                raise ValueError("source class must be a nonempty string")
            if key.birth_time is None:
                if key.entry_origin is not None:
                    raise ValueError("unaged source cannot have an entry origin")
            elif not np.isfinite(key.birth_time) or key.entry_origin not in ENTRY_ORIGINS:
                raise ValueError("finite forest source needs a birth and accepted origin")
        area = np.asarray(self.area)
        if area.ndim != 2 or area.shape[1] != len(keys):
            raise ValueError("area table must have region and source axes")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "area", _array(area, area.shape, "area table"))


@dataclass(frozen=True)
class SelectedArea:
    """Nonzero COO entries, ordered by (region, event, opening source index)."""
    region_index: np.ndarray
    event_index: np.ndarray
    source_index: np.ndarray
    area: np.ndarray

    def __post_init__(self):
        area = np.asarray(self.area)
        if area.ndim != 1 or np.any(area <= 0):
            raise ValueError("selected COO areas must be a positive vector")
        for name in ("region_index", "event_index", "source_index"):
            values = np.asarray(getattr(self, name))
            if values.shape != area.shape or values.dtype.kind not in "iu" or np.any(values < 0):
                raise ValueError("selected COO indices must be matching nonnegative integer vectors")
            object.__setattr__(self, name, _readonly(values, np.int64))
        object.__setattr__(self, "area", _array(area, area.shape, "selected area"))
        coordinates = np.column_stack((self.region_index, self.event_index, self.source_index))
        if len(np.unique(coordinates, axis=0)) != len(area):
            raise ValueError("duplicate selected COO coordinate")


@dataclass(frozen=True)
class RegionalPlan:
    """Pure labels and arrays; no carbon state references or member axis."""
    region_ids: tuple
    events: tuple
    strategy: str
    interval_start: int
    interval_end: int
    event_time: float
    environment_year: int
    next_year: int
    opening: AreaTable
    selected: SelectedArea
    closing: AreaTable
    requested_area: np.ndarray  # region, event
    executed_area: np.ndarray  # region, event
    surviving: AreaTable  # opening keys: old area left before incoming flows
    initial_land_area: np.ndarray | None = None  # fixed regional scale for numerical checks
    adjusted_native_sources: int = 0
    removed_native_requests: int = 0

    def __post_init__(self):
        regions, events = _ids(self.region_ids, "region IDs"), tuple(self.events)
        if len({e.event_id for e in events}) != len(events):
            raise ValueError("duplicate regional event IDs")
        if (any(type(value) is not int for value in
                (self.interval_start, self.interval_end, self.next_year, self.environment_year))
                or self.interval_end != self.interval_start + 1
                or self.next_year != self.interval_end or self.environment_year != self.interval_end
                or self.event_time != self.interval_start + .5 or self.strategy not in STRATEGIES):
            raise ValueError("invalid annual regional plan calendar or strategy")
        object.__setattr__(self, "region_ids", regions)
        object.__setattr__(self, "events", events)
        shape = (len(regions), len(events))
        for name in ("requested_area", "executed_area"):
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        initial = sum_rows(self.opening.area) if self.initial_land_area is None else self.initial_land_area
        object.__setattr__(self, "initial_land_area", _array(initial, (len(regions),), "initial regional land"))
        tolerance = NUMERICAL_RTOL * self.initial_land_area
        if any(table.area.shape[0] != len(regions) for table in
               (self.opening, self.closing, self.surviving)):
            raise ValueError("plan area region count differs")
        if set(self.surviving.keys) != set(self.opening.keys):
            raise ValueError("surviving area must retain exactly the opening source keys")
        for key in self.opening.keys:
            if key.birth_time is not None and key.birth_time > self.interval_start:
                raise ValueError("opening birth cannot be later than the annual boundary")
        for key in self.closing.keys:
            if key.birth_time is not None and key.birth_time > self.event_time:
                raise ValueError("closing birth cannot follow the annual event")
        flow = self.selected
        for values, size in ((flow.region_index, len(regions)), (flow.event_index, len(events)),
                             (flow.source_index, len(self.opening.keys))):
            if np.any(values >= size):
                raise ValueError("selected COO index outside coordinate bounds")
        source_rows, event_rows = {}, {}
        for r, e, k, amount in zip(flow.region_index, flow.event_index, flow.source_index, flow.area):
            if self.opening.keys[k].current_class != events[e].source_class:
                raise ValueError("selected source key differs from event source class")
            source_rows.setdefault((r, k), []).append(amount)
            event_rows.setdefault((r, e), []).append(amount)
        for (r, k), values in source_rows.items():
            total, supply = fsum(values), self.opening.area[r, k]
            if supply == 0 or total - supply > tolerance[r]:
                raise ValueError("selected regional source budget exceeds opening")
        survivor_index = {key: i for i, key in enumerate(self.surviving.keys)}
        for r in range(len(regions)):
            for k, key in enumerate(self.opening.keys):
                survivor = self.surviving.area[r, survivor_index[key]]
                supply = self.opening.area[r, k]
                partition = fsum((*source_rows.get((r, k), ()), survivor))
                if abs(partition-supply) > tolerance[r]:
                    raise ValueError("selected plus surviving source partition differs from opening")
        executed = np.zeros(shape)
        for (r, e), values in event_rows.items():
            executed[r, e] = fsum(values)
        if (np.any(abs(executed - self.executed_area) > tolerance[:, None])
                or np.any(abs(executed - self.requested_area) > tolerance[:, None])):
            raise ValueError("regional event totals differ from selected area")

    def validate_classes(self, classes):
        """Validate prepared roles and routing, without data repair or carbon."""
        classes = tuple(classes)
        if getattr(self, "_validated_classes", None) == classes:
            return
        lookup = {c.class_id: c for c in classes}
        if len(lookup) != len(classes):
            raise ValueError("duplicate class IDs")
        fixed = {SourceKey(c.class_id) for c in classes if c.role != "forest_cohort"}
        for table in (self.opening, self.surviving, self.closing):
            if not fixed.issubset(table.keys):
                raise ValueError("fixed primary/nonforest keys missing")
            for key in table.keys:
                if key.current_class not in lookup:
                    raise ValueError("unknown source key class")
                finite = lookup[key.current_class].role == "forest_cohort"
                if finite != (key.birth_time is not None):
                    raise ValueError("source birth/origin does not match class role")
        for i, event in enumerate(self.events):
            if event.source_class not in lookup or event.target_class not in lookup:
                raise ValueError("event class absent from prepared class table")
            if np.any(self.executed_area[:, i]):
                _validate_spec(event, lookup[event.source_class], lookup[event.target_class])
        # Reconstruct destinations separately from surviving opening amounts.
        inflows = {}
        f = self.selected
        for r, e, k, amount in zip(f.region_index, f.event_index, f.source_index, f.area):
            key, event = self.opening.keys[k], self.events[e]
            if lookup[event.target_class].role == "nonforest":
                dest = SourceKey(event.target_class)
            elif event.age_action == "keep" or secondary_stand_harvest(event):
                dest = SourceKey(event.target_class, key.birth_time, key.entry_origin)
            else:
                origin = "cleared" if event.vegetation_action == "clear" else event.source_class
                dest = SourceKey(event.target_class, self.event_time, origin)
            inflows.setdefault((r, dest), []).append(amount)
        opening = {k: i for i, k in enumerate(self.opening.keys)}
        surviving = {k: i for i, k in enumerate(self.surviving.keys)}
        closing = {k: i for i, k in enumerate(self.closing.keys)}
        keys = set(opening) | set(closing) | {k for _, k in inflows}
        for r in range(len(self.region_ids)):
            for key in keys:
                survivor = self.surviving.area[r, surviving[key]] if key in surviving else 0.
                incoming = fsum(inflows.get((r, key), ()))
                expected = fsum((survivor, incoming))
                actual = self.closing.area[r, closing[key]] if key in closing else 0.
                scale = max(survivor, incoming, expected, actual)
                if abs(expected-actual) > NUMERICAL_RTOL * self.initial_land_area[r]:
                    raise ValueError("regional closing differs from selected source routing")
        # All arrays and definitions are immutable; paired trajectories and
        # member batches share this one prepared-plan check.
        object.__setattr__(self, "_validated_classes", classes)

    @property
    def area_roundoff(self):
        return self.executed_area - self.requested_area


class AreaShortage(ValueError):
    """Frozen opening cannot pay the full annual source budget."""


def _check_supply(demand, supply, tolerance, message):
    bad = demand - supply > tolerance
    if np.any(bad):
        cell = int(np.flatnonzero(bad)[0])
        raise AreaShortage(f"{message}, batch cell={cell}, demand={demand[cell]:.17g}, supply={supply[cell]:.17g}")


def _proportional(area, demand, tolerance):
    supply = sum_rows(area)
    _check_supply(demand, supply, tolerance, "proportional selection exceeds source")
    fraction = np.divide(demand, supply, out=np.zeros_like(demand), where=supply > 0)
    # Only a float64 tail can exceed one, after the supply check above.
    return area * np.minimum(fraction, 1.)[:, None]


def _select(area, births, demand, mode, tolerance):
    if mode == "proportional":
        return _proportional(area, demand, tolerance)
    _check_supply(demand, sum_rows(area), tolerance, "directional selection exceeds source")
    out = np.zeros_like(area)
    remaining = demand.copy()
    ages = np.unique(births)
    if mode == "youngest":
        ages = ages[::-1]
    for birth in ages:
        at = births == birth
        supply = sum_rows(area[:, at])
        take = np.minimum(remaining, supply)
        out[:, at] = _proportional(area[:, at], take, tolerance)
        remaining -= take
    return out


def _reduce(area, weights):
    # Contiguous reduction along cells uses NumPy pairwise summation.
    transposed = np.ascontiguousarray(area.T)
    return np.stack([(transposed * weights[:, r]).sum(axis=1)
                     for r in range(weights.shape[1])])


def _readonly(value, dtype=None):
    value = np.array(value, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


def advance_geometry(state, events, event_area_Mha, *, year, region_ids, region_fraction):
    """Allocate a complete annual batch and return (new state, regional plan).

    Input event areas have shape (cell, event), weights (cell, region). Events
    are sorted by ID with their demand columns. For the proportional strategy
    all harvest modes become proportional; the specific strategy uses the
    prepared modes (e.g. secmf oldest, secyf youngest). Zero unresolved events
    remain dormant. No state is changed, even if validation fails late.
    """
    if type(year) is not int or year != state.next_year:
        raise ValueError("annual geometry cursor mismatch")
    region_ids = _ids(region_ids, "region IDs")
    cells, keys = state.area.shape
    weights = _array(region_fraction, (cells, len(region_ids)), "region fractions")
    weight_sum = sum_rows(weights)
    if np.any(np.abs(weight_sum - 1.) > _roundoff(weight_sum, 1.)):
        raise ValueError("region fractions must sum to one for each cell")
    events = tuple(events)
    if len({e.event_id for e in events}) != len(events):
        raise ValueError("duplicate annual event ID")
    demand = _array(event_area_Mha, (cells, len(events)), "event area")
    order = sorted(range(len(events)), key=lambda i: events[i].event_id)
    events, demand = tuple(events[i] for i in order), demand[:, order]
    classes = {c.class_id: c for c in state.classes}
    for j, event in enumerate(events):
        if event.source_class not in classes or event.target_class not in classes:
            raise ValueError("event references an unknown class")
        if np.any(demand[:, j]):
            _validate_spec(event, classes[event.source_class], classes[event.target_class])

    # Small representational differences are paid from actual opening area.
    # Original requests stay unchanged in the plan and all reductions accumulate.
    tolerance = NUMERICAL_RTOL * state.initial_land_area
    original = demand.copy()
    adjusted_sources = 0
    for source in classes:
        ix = [j for j, e in enumerate(events) if e.source_class == source]
        at = [i for i, k in enumerate(state.keys) if k.current_class == source]
        requested = sum_rows(original[:, ix])
        supply = sum_rows(state.area[:, at])
        _check_supply(requested, supply, tolerance, f"year={year}, source={source}")
        short = requested > supply
        adjusted_sources += int(np.count_nonzero(short))
        factor = np.divide(supply, requested, out=np.ones(cells), where=short)
        demand[:, ix] = original[:, ix] * factor[:, None]
        # A common downward ULP handles multiplication tails without changing
        # relative event priorities or inserting area into a cohort.
        over = sum_rows(demand[:, ix]) > supply
        while np.any(over):
            factor[over] = np.nextafter(factor[over], 0.)
            demand[:, ix] = original[:, ix] * factor[:, None]
            over = sum_rows(demand[:, ix]) > supply

    survivors = state.area.copy()
    incoming_area = {}
    executed_native = np.zeros_like(demand)
    flow_region, flow_event, flow_source, flow_area = [], [], [], []

    def emit(event_index, slots, take):
        if not np.any(take):
            return
        event = events[event_index]
        executed_native[:, event_index] += sum_rows(take)
        reduced = _reduce(take, weights)
        region, local_key = np.nonzero(reduced)
        flow_region.extend(region.tolist())
        flow_event.extend([event_index] * len(region))
        flow_source.extend(slots[local_key].tolist())
        flow_area.extend(reduced[region, local_key].tolist())
        target_role = classes[event.target_class].role
        if target_role == "nonforest":
            incoming = [(SourceKey(event.target_class), sum_rows(take))]
        elif event.age_action == "keep" or secondary_stand_harvest(event):
            incoming = [(SourceKey(event.target_class, state.keys[slot].birth_time,
                                   state.keys[slot].entry_origin), take[:, j])
                        for j, slot in enumerate(slots)]
        else:
            origin = "cleared" if event.vegetation_action == "clear" else event.source_class
            incoming = [(SourceKey(event.target_class, year + .5, origin), sum_rows(take))]
        for key, amount in incoming:
            if key not in incoming_area:
                incoming_area[key] = np.zeros(cells)
            incoming_area[key] += amount

    def distribute(selected, group, slots):
        if not group:
            return
        demands = demand[:, group]
        total = sum_rows(demands)
        tail = np.argmax(demands, axis=1)  # Stable event-ID tie break.
        remainder = selected.copy()
        for local, event_index in enumerate(group):
            fraction = np.divide(demand[:, event_index], total,
                                 out=np.zeros(cells), where=total > 0)
            take = selected * fraction[:, None]
            take[tail == local] = 0.
            # Guard arithmetic tails only, never erase a positive small cohort.
            excess = take - remainder
            if np.any(excess > tolerance[:, None]):
                raise ValueError("floating-point event partition failed")
            take = np.minimum(take, remainder)
            remainder -= take
            emit(event_index, slots, take)
        for local, event_index in enumerate(group):
            take = np.where((tail == local)[:, None], remainder, 0.)
            emit(event_index, slots, take)

    for source in classes:
        slots = np.array([i for i, k in enumerate(state.keys) if k.current_class == source], dtype=int)
        group = [j for j, e in enumerate(events) if e.source_class == source and np.any(demand[:, j])]
        if not group:
            continue
        available = state.area[:, slots]
        conversions = [j for j in group if events[j].kind == "conversion"]
        conversion_take = _proportional(available, sum_rows(demand[:, conversions]), tolerance)
        residual = available - conversion_take
        distribute(conversion_take, conversions, slots)
        harvests = [j for j in group if events[j].kind == "harvest"]
        modes = {j: ("proportional" if state.strategy == "proportional" else events[j].selection)
                 for j in harvests}
        if "proportional" in modes.values() and len(set(modes.values())) > 1:
            raise ValueError("mixed proportional/directional harvest needs an explicit policy")
        births = np.array([state.keys[i].birth_time if state.keys[i].birth_time is not None
                           else year for i in slots])
        for mode in ("proportional", "oldest", "youngest"):
            tied = [j for j in harvests if modes[j] == mode]
            if not tied:
                continue
            take = _select(residual, births, sum_rows(demand[:, tied]), mode, tolerance)
            distribute(take, tied, slots)
            residual -= take
        survivors[:, slots] = residual

    if np.any(np.abs(executed_native - demand) > tolerance[:, None]):
        raise ValueError("event area was not fully reserved")
    # Merge surviving old slots with all incoming keys, including keep into
    # an existing target. Separate accumulation avoids cancelling tiny inflows
    # against an opening slot which was completely removed this year.
    closing = {key: survivors[:, i].copy() for i, key in enumerate(state.keys)}
    for key, amount in incoming_area.items():
        if key not in closing:
            closing[key] = np.zeros(cells)
        closing[key] += amount
    signed_difference = executed_native-original
    if np.any(sum_rows(np.maximum(signed_difference, 0.)) > tolerance):
        raise ValueError("Numerical overpayment exceeds the fixed initial-land tolerance")
    reduction = state.numerical_area_reduction + sum_rows(np.maximum(-signed_difference, 0.))
    out = GeometryState(state.classes, state.cell_ids, tuple(closing),
                        np.column_stack(list(closing.values())), year + 1, state.strategy,
                        state.initial_land_area, reduction)
    net = np.zeros_like(state.class_area)
    for j,e in enumerate(events):
        if e.source_class != e.target_class:
            net[:, [c.class_id for c in state.classes].index(e.source_class)] -= executed_native[:,j]
            net[:, [c.class_id for c in state.classes].index(e.target_class)] += executed_native[:,j]
    if np.any(np.abs(out.class_area-state.class_area-net) > tolerance[:,None]):
        raise ValueError("Actual class net flow exceeds the fixed initial-land tolerance")
    before, after = sum_rows(state.area), sum_rows(out.area)
    if np.any(np.abs(after-before) > tolerance):
        raise ValueError("Annual land balance exceeds the fixed initial-land tolerance")

    # An event can be emitted twice for different cells; coalesce its regional
    # entries once, without constructing event x birth x cell arrays.
    if flow_area:
        coordinates = np.array([flow_region, flow_event, flow_source], dtype=np.int64).T
        coordinates, inverse = np.unique(coordinates, axis=0, return_inverse=True)
        values = np.zeros(len(coordinates))
        np.add.at(values, inverse, flow_area)
    else:
        coordinates, values = np.empty((0, 3), dtype=np.int64), np.empty(0)
    selected = SelectedArea(*(_readonly(coordinates[:, i]) for i in range(3)), _readonly(values))
    executed = np.zeros((len(region_ids), len(events)))
    # Pairwise sums of contiguous event runs keep the regional audit at the
    # same float64 precision, even when an event spans many birth keys.
    if len(values):
        starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(coordinates[:, :2], axis=0), axis=1))]
        ends = np.r_[starts[1:], len(values)]
        for start, end in zip(starts, ends):
            r, e = coordinates[start, :2]
            executed[r, e] = values[start:end].sum()
    requested = _reduce(original, weights)
    regional_initial = _reduce(state.initial_land_area[:,None], weights)[:,0]
    if np.any(np.abs(executed - requested) > NUMERICAL_RTOL * regional_initial[:,None]):
        raise ValueError("regional event reduction failed its area balance")
    plan = RegionalPlan(region_ids, events, state.strategy, year, year + 1, year + .5,
                        year + 1, year + 1,
                        AreaTable(state.keys, _readonly(_reduce(state.area, weights))), selected,
                        AreaTable(out.keys, _readonly(_reduce(out.area, weights))),
                        _readonly(requested), _readonly(executed),
                        AreaTable(state.keys, _readonly(_reduce(survivors, weights))), regional_initial,
                        adjusted_sources, int(np.count_nonzero((original > 0) & (demand == 0))))
    return out, plan

"""Route already selected regional source areas through actual carbon stocks.

Geometry alone chooses land. All events use one frozen pre-event live/dead
state; regional nonforest and biome dead pools retain their mixing convention.
"""
from dataclasses import dataclass
from functools import cached_property
from math import fsum
import numpy as np

from .forest_age import TwoPoolForestState, _carbon_array, _member_ids, align_members
from .land_geometry import (LandClass, EventSpec, SourceKey, RegionalPlan,
                            ENTRY_ORIGINS, _key_order, _roundoff, secondary_stand_harvest)


def _ids(values, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values) or any(not isinstance(x, str) or not x for x in values):
        raise ValueError(f"{name} must contain unique nonempty strings")
    return values


@dataclass(frozen=True)
class LandEventState:
    forest: TwoPoolForestState
    classes: tuple
    biome_ids: tuple
    product_ids: tuple
    cohort_class: tuple
    nonforest_area: np.ndarray
    nonforest_carbon: np.ndarray
    cwd: np.ndarray
    soil: np.ndarray
    hwp: np.ndarray
    next_event_year: int

    def __post_init__(self):
        object.__setattr__(self, "classes", tuple(self.classes))
        _ids([x.class_id for x in self.classes], "class IDs")
        for name in ("biome_ids", "product_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if sum(x.role == "forest_unaged" for x in self.classes) != 1:
            raise ValueError("exactly one unaged forest class is supported")
        if any(x.biome not in self.biome_ids for x in self.classes):
            raise ValueError("unknown biome in class table")
        forest_biomes = {x.biome for x in self.classes if x.role != "nonforest"}
        if len(forest_biomes) != 1:
            raise ValueError("forest roles must share one forest biome")
        if any(x.biome in forest_biomes for x in self.classes if x.role == "nonforest"):
            raise ValueError("nonforest and forest roles cannot share a biome")
        object.__setattr__(self, "cohort_class", tuple(self.cohort_class))
        allowed = {x.class_id for x in self.classes if x.role == "forest_cohort"}
        if len(self.cohort_class) != self.forest.secondary_area.size or any(x not in allowed for x in self.cohort_class):
            raise ValueError("cohort classes must match forest cohort slots")
        m, n, b, p = len(self.forest.member_ids), len(self.nonforest_classes), len(self.biome_ids), len(self.product_ids)
        for name, shape in (("nonforest_area", (n,)), ("nonforest_carbon", (m, n)),
                            ("cwd", (m, b)), ("soil", (m, b)), ("hwp", (m, b, p))):
            object.__setattr__(self, name, _carbon_array(getattr(self, name), shape, name))
        if np.any(self.nonforest_carbon[:, self.nonforest_area == 0] != 0):
            raise ValueError("zero nonforest class area cannot contain live carbon")
        if np.any(self.cwd[:, self.biome_area == 0] != 0) or np.any(self.soil[:, self.biome_area == 0] != 0):
            raise ValueError("zero biome area cannot contain CWD or soil")
        if type(self.next_event_year) is not int:
            raise ValueError("next_event_year must be an integer calendar year")
        if len(set(self.source_keys)) != len(self.source_keys):
            raise ValueError("duplicate complete forest source key")

    @property
    def nonforest_classes(self):
        return tuple(x.class_id for x in self.classes if x.role == "nonforest")

    @property
    def source_classes(self):
        primary = next(x.class_id for x in self.classes if x.role == "forest_unaged")
        return (primary,) + self.cohort_class + self.nonforest_classes

    @property
    def source_keys(self):
        primary = next(x.class_id for x in self.classes if x.role == "forest_unaged")
        return ((SourceKey(primary),) + tuple(SourceKey(c, b, o) for c, b, o in
                zip(self.cohort_class, self.forest.birth_time, self.forest.entry_origin))
                + tuple(SourceKey(c) for c in self.nonforest_classes))

    @property
    def source_area(self):
        return np.concatenate(([self.forest.primary_area], self.forest.secondary_area, self.nonforest_area))

    @cached_property
    def area_class(self):
        buckets = {item.class_id: [] for item in self.classes}
        for area, source in zip(self.source_area, self.source_classes):
            buckets[source].append(area)
        value = np.array([fsum(buckets[item.class_id]) for item in self.classes])
        value.setflags(write=False)
        return value

    @cached_property
    def biome_area(self):
        area = self.area_class
        value = np.array([fsum(a for a, c in zip(area, self.classes) if c.biome == b) for b in self.biome_ids])
        value.setflags(write=False)
        return value

    @property
    def total_carbon(self):
        return (self.forest.total_carbon + self.nonforest_carbon.sum(axis=1)
                + self.cwd.sum(axis=1) + self.soil.sum(axis=1) + self.hwp.sum(axis=(1, 2)))

    def snapshot(self):
        return dict(schema="land-events-v2", forest=self.forest.snapshot(),
                    classes=[vars(x) for x in self.classes], biome_ids=list(self.biome_ids),
                    product_ids=list(self.product_ids), cohort_class=list(self.cohort_class),
                    nonforest_area=self.nonforest_area.tolist(), nonforest_carbon=self.nonforest_carbon.tolist(),
                    cwd=self.cwd.tolist(), soil=self.soil.tolist(), hwp=self.hwp.tolist(), next_event_year=self.next_event_year)

    @classmethod
    def from_snapshot(cls, data):
        data = dict(data)
        if data.pop("schema") != "land-events-v2":
            raise ValueError("unsupported land-event snapshot schema")
        data["forest"] = TwoPoolForestState.from_snapshot(data["forest"])
        data["classes"] = tuple(LandClass(**x) for x in data["classes"])
        return cls(**data)


@dataclass(frozen=True)
class EventRouting:
    member_ids: tuple
    biome_ids: tuple
    product_ids: tuple
    p_wood: np.ndarray
    p_hwp: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "member_ids", _member_ids(self.member_ids))
        for name in ("biome_ids", "product_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        m, b, p = len(self.member_ids), len(self.biome_ids), len(self.product_ids)
        object.__setattr__(self, "p_wood", _carbon_array(self.p_wood, (m, b), "p_wood"))
        object.__setattr__(self, "p_hwp", _carbon_array(self.p_hwp, (m, b, p), "p_hwp"))
        if np.any(self.p_wood > 1) or np.any(self.p_hwp.sum(axis=2) > 1):
            raise ValueError("wood/product shares exceed one")


@dataclass(frozen=True)
class EventLedger:
    event_ids: tuple
    requested_area: np.ndarray
    executed_area: np.ndarray
    removed: np.ndarray       # member, W/F, event; nonforest split is assumed
    retained: np.ndarray     # member, W/F, event
    cwd_added: np.ndarray    # member, event; destination biome
    soil_added: np.ndarray   # member, event; destination biome
    hwp_added: np.ndarray    # member, event, product; source biome
    dead_transferred: np.ndarray  # member, CWD/soil, event; pre-event stocks
    budget_residual: np.ndarray

    @property
    def area_roundoff(self):
        """Signed executed-requested residual; numerical partition only."""
        return self.executed_area-self.requested_area


def apply_events(state, plan, routing):
    """Consume one RegionalPlan once; return (new state, physical EventLedger)."""
    if not isinstance(plan, RegionalPlan):
        raise TypeError("carbon routing requires a RegionalPlan")
    if state.next_event_year != plan.interval_start or state.forest.time != plan.event_time:
        raise ValueError("annual event cursor or pre-event clock differs from plan")
    plan.validate_classes(state.classes)
    if state.forest.region_id not in plan.region_ids:
        raise ValueError("state region absent from geometry plan")
    if routing.biome_ids != state.biome_ids or routing.product_ids != state.product_ids:
        raise ValueError("routing biome/product coordinates differ")
    ri = plan.region_ids.index(state.forest.region_id)
    state_keys, available = state.source_keys, state.source_area
    source_index = {key: i for i, key in enumerate(state_keys)}
    plan_index = {key: i for i, key in enumerate(plan.opening.keys)}
    for key in set(source_index) | set(plan_index):
        actual = available[source_index[key]] if key in source_index else 0.
        expected = plan.opening.area[ri, plan_index[key]] if key in plan_index else 0.
        if abs(actual-expected) > _roundoff(actual, expected):
            raise ValueError("complete-key opening differs from actual state")
    # Geometry owns area. After the precision check, bind the unchanged actual
    # stocks to its canonical opening instead of perpetuating a differently
    # grouped carbon-side area sum. No carbon or selected flow is altered.
    available = np.array([plan.opening.area[ri, plan_index[key]] if key in plan_index else 0.
                          for key in state_keys])
    pwood = align_members(routing.p_wood, routing.member_ids, state.forest.member_ids)
    phwp = align_members(routing.p_hwp, routing.member_ids, state.forest.member_ids)
    classes = {x.class_id: x for x in state.classes}
    biome_index = {b: i for i, b in enumerate(state.biome_ids)}
    nf_index = {c: i for i, c in enumerate(state.nonforest_classes)}
    m, e, k = len(state.forest.member_ids), len(plan.events), state.forest.secondary_area.size
    flow = plan.selected
    rows = np.flatnonzero(flow.region_index == ri)
    by_event = [[] for _ in plan.events]
    by_source = [[] for _ in state_keys]
    for row in rows:
        key = plan.opening.keys[flow.source_index[row]]
        if key not in source_index or available[source_index[key]] == 0:
            raise ValueError("positive selected area has no actual source")
        slot, event, amount = source_index[key], int(flow.event_index[row]), flow.area[row]
        by_event[event].append((slot, amount))
        by_source[slot].append(amount)
    totals = np.array([fsum(values) for values in by_source])
    if np.any(totals-available > _roundoff(totals, available)):
        raise ValueError("actual source budget exceeded")
    survivor_index = {key: i for i, key in enumerate(plan.surviving.keys)}
    remaining_area = np.array([plan.surviving.area[ri, survivor_index[key]]
                               if key in survivor_index else 0. for key in state_keys])
    # Partition the same stock among explicit selected parcels and old survivors.
    # Independent regional sums can differ by ulps; validate that identity first,
    # then use the sum of these parts, not an opening-minus-outflow subtraction.
    partition_area = np.array([fsum((*values, survivor))
                               for values, survivor in zip(by_source, remaining_area)])
    if np.any(abs(partition_area-available) > _roundoff(partition_area, available)):
        raise ValueError("actual source partition differs from geometry opening")
    fraction_left = np.divide(remaining_area, partition_area, out=np.zeros_like(available), where=partition_area > 0)
    forest_carbon = np.concatenate((state.forest.primary_carbon[:, :, None], state.forest.secondary_carbon), axis=2)
    forest_left = forest_carbon * fraction_left[None, None, :k+1]
    nf_carbon = state.nonforest_carbon * fraction_left[None, k+1:]
    nf_area = remaining_area[k+1:].copy()
    # Keep old survivors separate from all incoming stocks/areas.
    cohorts = {state_keys[j+1]: (remaining_area[j+1], forest_left[:, :, j+1].copy()) for j in range(k)}
    incoming = {}
    removed, retained = np.zeros((m, 2, e)), np.zeros((m, 2, e))
    cwd_added, soil_added = np.zeros((m, e)), np.zeros((m, e))
    hwp_added = np.zeros((m, e, len(state.product_ids)))
    dead_moved = np.zeros((m, 2, e))
    cwd, soil, hwp = state.cwd.copy(), state.soil.copy(), state.hwp.copy()
    executed = plan.executed_area[ri]
    dead_left = [[] for _ in state.biome_ids]
    dead_out = [[] for _ in state.biome_ids]
    for key, survivor in zip(state_keys, remaining_area):
        dead_left[biome_index[classes[key.current_class].biome]].append(survivor)
    for i, spec in enumerate(plan.events):
        sb = biome_index[classes[spec.source_class].biome]
        cross = spec.kind == "conversion" and classes[spec.source_class].biome != classes[spec.target_class].biome
        # Use the same selected source parts as the live routing. Existing dead
        # carbon is retained for all parcels which remain in their source biome.
        (dead_out if cross else dead_left)[sb].extend(amount for _, amount in by_event[i])
    dead_left_area = np.array([fsum(values) for values in dead_left])
    da = np.array([fsum((*left, *out)) for left, out in zip(dead_left, dead_out)])
    actual_biome_area = np.array([fsum(area for key, area in zip(state_keys, available)
                                     if classes[key.current_class].biome == biome) for biome in state.biome_ids])
    if np.any(abs(da-actual_biome_area) > _roundoff(da, actual_biome_area)):
        raise ValueError("dead-stock source partition differs from geometry opening")
    dead_fraction_left = np.divide(dead_left_area, da, out=np.zeros_like(da), where=da > 0)
    cwd *= dead_fraction_left[None, :]
    soil *= dead_fraction_left[None, :]

    def add_forest(key, area, carbon):
        if key in incoming:
            old_area, old_carbon = incoming[key]
            incoming[key] = (old_area+area, old_carbon+carbon)
        else:
            incoming[key] = (area, carbon.copy())

    for i, spec in enumerate(plan.events):
        area = executed[i]
        if not area:
            continue
        source, target = classes[spec.source_class], classes[spec.target_class]
        sb, tb = biome_index[source.biome], biome_index[target.biome]
        for j, amount in by_event[i]:
            fraction = amount/partition_area[j]
            if source.role == "nonforest":
                carbon = state.nonforest_carbon[:, nf_index[source.class_id]] * fraction
                taken = np.stack((pwood[:, sb]*carbon, (1-pwood[:, sb])*carbon), axis=1)
            else:
                taken = forest_carbon[:, :, j] * fraction
            removed[:, :, i] += taken
            if target.role == "forest_cohort" and spec.age_action == "keep":
                key = state_keys[j]
                add_forest(SourceKey(target.class_id, key.birth_time, key.entry_origin), amount, taken)
            elif secondary_stand_harvest(spec):
                # Same fraction of stock leaves once. The stand keeps its area and birth.
                key = state_keys[j]
                add_forest(SourceKey(target.class_id, key.birth_time, key.entry_origin), amount,
                           np.zeros_like(taken))
        if spec.vegetation_action == "retain":
            retained[:, :, i] = removed[:, :, i]
        else:
            hwp_added[:, i, :] = removed[:, 0, i, None] * phwp[:, sb, :]
            cwd_added[:, i] = removed[:, 0, i] * (1-phwp[:, sb, :].sum(axis=1))
            soil_added[:, i] = removed[:, 1, i]
            cwd[:, tb] += cwd_added[:, i]
            soil[:, tb] += soil_added[:, i]
            hwp[:, sb, :] += hwp_added[:, i, :]
        if target.role == "nonforest":
            n = nf_index[target.class_id]
            nf_area[n] += area
            nf_carbon[:, n] += retained[:, :, i].sum(axis=1)
        elif spec.age_action != "keep" and not secondary_stand_harvest(spec):
            origin = "cleared" if spec.vegetation_action == "clear" else spec.source_class
            add_forest(SourceKey(target.class_id, plan.event_time, origin), area, retained[:, :, i])
        if spec.kind == "conversion":
            dead_moved[:, 0, i] = state.cwd[:, sb] * (area/da[sb])
            dead_moved[:, 1, i] = state.soil[:, sb] * (area/da[sb])
            if sb != tb:
                cwd[:, tb] += dead_moved[:, 0, i]
                soil[:, tb] += dead_moved[:, 1, i]
    for key, (area, carbon) in incoming.items():
        old_area, old_carbon = cohorts.get(key, (0., np.zeros((m, 2))))
        cohorts[key] = (old_area+area, old_carbon+carbon)
    closing = dict(zip(plan.closing.keys, plan.closing.area[ri]))
    reconstructed = {state_keys[0]: remaining_area[0]}
    reconstructed.update({key: area for key, (area, _) in cohorts.items()})
    reconstructed.update({SourceKey(name): area for name, area in zip(state.nonforest_classes, nf_area)})
    for key in set(reconstructed) | set(closing):
        actual, expected = reconstructed.get(key, 0.), closing.get(key, 0.)
        opening = available[source_index[key]] if key in source_index else 0.
        scale = max(actual, expected, opening)
        if abs(actual-expected) > _roundoff(scale, scale):
            raise ValueError("complete-key closing differs from geometry")
    # Keep exactly the geometry's closing areas. A zero-area stock is rejected
    # by the state constructor, never silently discarded by a smallness cutoff.
    cohorts = {key: (closing.get(key, 0.), carbon) for key, (_, carbon) in cohorts.items()}
    nf_area = np.array([closing[SourceKey(name)] for name in state.nonforest_classes])
    class_ids = tuple(c.class_id for c in state.classes)
    keys = sorted((key for key, (area, carbon) in cohorts.items() if area != 0 or np.any(carbon != 0)),
                  key=lambda key: _key_order(key, class_ids))
    stocks = np.stack([cohorts[key][1] for key in keys], axis=2) if keys else np.empty((m, 2, 0))
    forest = TwoPoolForestState(state.forest.time, state.forest.member_ids, closing[state_keys[0]],
                               forest_left[:, :, 0], [key.birth_time for key in keys],
                               [cohorts[key][0] for key in keys], stocks, state.forest.region_id,
                               tuple(key.entry_origin for key in keys))
    out = LandEventState(forest, state.classes, state.biome_ids, state.product_ids,
                         tuple(key.current_class for key in keys), nf_area, nf_carbon, cwd, soil, hwp,
                         plan.next_year)
    residual = out.total_carbon-state.total_carbon
    if np.any(np.abs(residual) > 1e-10*np.maximum(1., state.total_carbon)):
        raise ValueError("event carbon budget failed; no state committed")
    ledger = EventLedger(tuple(spec.event_id for spec in plan.events), plan.requested_area[ri],
                         executed, removed, retained, cwd_added, soil_added, hwp_added, dead_moved, residual)
    return out, ledger

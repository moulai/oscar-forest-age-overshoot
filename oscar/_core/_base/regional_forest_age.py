"""Regional annual forest cohorts and their actual carbon transfers.

This physical kernel uses ordinary xarray state variables. It does not read
rasters, construct spatial event plans, run a second climate model, or retain
historical source labels. Units are Mha, PgC and years.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from functools import cached_property
import numpy as np
import xarray as xr

from .physical_land import advance_live_carbon
from .dead_carbon import DeadCarbonRates, advance_dead_carbon
from .land_geometry import secondary_stand_harvest
from .regional_forest_age_allocation import select_joint_area, select_joint_block, select_ranked_area
from .regional_forest_age_diagnostics import AGE_SUPPORT, LAND_ACTION, age_support_index, event_age_summary

STATE_NAMES = ("rf_A_primary", "rf_W_primary", "rf_F_primary", "rf_A_forest",
               "rf_W_forest", "rf_F_forest", "rf_A_nonforest", "rf_C_nonforest",
               "rf_C_cwd", "rf_C_soil", "rf_C_hwp")


@dataclass(frozen=True)
class RegionalLayout:
    """Fixed native coordinates and prepared land/event semantics.

    classes expose class_id, biome and role; events expose event_id,
    source_class, target_class, kind, vegetation_action and age_action.
    These are metadata, not a grid plan. Births are physical year midpoints.
    """
    regions: tuple
    members: tuple
    classes: tuple
    births: tuple
    biomes: tuple
    products: tuple
    events: tuple = ()
    source_members: tuple | None = None

    def __post_init__(self):
        for field in ("regions", "members", "biomes", "products"):
            values = getattr(self, field)
            if not values or len(set(values)) != len(values):
                raise ValueError(field + " must contain unique coordinates")
        if len({c.class_id for c in self.classes}) != len(self.classes):
            raise ValueError("duplicate land classes")
        if self.source_members is not None and (len(self.source_members) != len(self.members)
                or len(set(self.source_members)) != len(self.members)):
            raise ValueError("source member identities must match the native config axis")
        if sum(c.role == "forest_unaged" for c in self.classes) != 1:
            raise ValueError("exactly one primary forest class is required")
        if not self.forest_classes or not self.nonforest_classes:
            raise ValueError("cohort forest and nonforest source classes are required")
        if len(set(self.births)) != len(self.births) or not np.all(np.diff(self.births) > 0):
            raise ValueError("birth coordinates must be unique and increasing")
        if any(not np.isfinite(b) or b % 1 != .5 for b in self.births):
            raise ValueError("annual birth coordinates must be year midpoints")
        if any(c.biome not in self.biomes for c in self.classes):
            raise ValueError("unknown biome")
        if any(c.biome != self.biomes[0] for c in self.classes if c.role != "nonforest"):
            raise ValueError("the first biome must represent all forest sources")
        names = {c.class_id: c for c in self.classes}
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("duplicate events")
        for e in self.events:
            if e.source_class not in names or e.target_class not in names:
                raise ValueError("event references an unknown class")
            source, target = names[e.source_class], names[e.target_class]
            if e.kind not in ("conversion", "harvest"):
                raise ValueError("unknown event kind")
            if e.vegetation_action not in ("clear", "retain", "unresolved"):
                raise ValueError("unknown vegetation action")
            if target.role == "forest_unaged":
                raise ValueError("events cannot create unaged primary forest")
            if target.role == "forest_cohort" and e.vegetation_action != "unresolved":
                if e.vegetation_action == "clear" and e.age_action != "reset":
                    raise ValueError("forest clearance must reset age")
                if e.vegetation_action == "retain":
                    if source.role == "forest_unaged":
                        raise ValueError("retained unaged-to-cohort conversion has no birth time")
                    expected = "keep" if source.role == "forest_cohort" else "reset"
                    if e.age_action != expected:
                        raise ValueError("retained vegetation has inconsistent age action")
            elif target.role == "nonforest" and e.age_action != "none":
                raise ValueError("nonforest target cannot carry an age action")

    @property
    def parameter_ids(self):
        return self.members if self.source_members is None else self.source_members

    @property
    def primary_class(self):
        return next(c.class_id for c in self.classes if c.role == "forest_unaged")

    @property
    def forest_classes(self):
        return tuple(c.class_id for c in self.classes if c.role == "forest_cohort")

    @property
    def nonforest_classes(self):
        return tuple(c.class_id for c in self.classes if c.role == "nonforest")

    @cached_property
    def event_index_groups(self):
        """Event rows and source columns in the order selection walks them."""
        classes = {c.class_id: c for c in self.classes}
        slots = self.slots
        groups = []
        for kind in ("conversion", "harvest"):
            for source in classes:
                rows = [i for i, ev in enumerate(self.events) if ev.kind == kind and ev.source_class == source]
                if not rows:
                    continue
                cols = [i for i, (c, _) in enumerate(slots) if c == source]
                groups.append((kind, source, np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)))
        return tuple(groups)

    @cached_property
    def slots(self):
        return ((self.primary_class, None),) + tuple((c, b) for c in self.forest_classes for b in self.births) + tuple((c, None) for c in self.nonforest_classes)

    @cached_property
    def _validation_template(self):
        """Coordinate metadata shared by state allocation and validation."""
        template = self.empty_state(0.)
        return ({name: template[name].dims for name in STATE_NAMES},
                {name: np.array(template[name].values) for name in template.dims})

    def empty_state(self, year):
        """Allocate explicit zero stocks; callers supply physical initial stocks."""
        coords = dict(config=list(self.members), reg_land=list(self.regions),
                      forest_class=list(self.forest_classes), birth_time=list(self.births),
                      nonforest_class=list(self.nonforest_classes), bio_land=list(self.biomes),
                      box_hwp=list(self.products))
        dims = {
            "rf_A_primary": ("config", "reg_land"),
            "rf_W_primary": ("config", "reg_land"), "rf_F_primary": ("config", "reg_land"),
            "rf_A_forest": ("config", "reg_land", "forest_class", "birth_time"),
            "rf_W_forest": ("config", "reg_land", "forest_class", "birth_time"),
            "rf_F_forest": ("config", "reg_land", "forest_class", "birth_time"),
            "rf_A_nonforest": ("config", "reg_land", "nonforest_class"),
            "rf_C_nonforest": ("config", "reg_land", "nonforest_class"),
            "rf_C_cwd": ("config", "reg_land", "bio_land"),
            "rf_C_soil": ("config", "reg_land", "bio_land"),
            "rf_C_hwp": ("config", "reg_land", "bio_land", "box_hwp"),
        }
        state = xr.Dataset({name: (axes, np.zeros(tuple(len(coords[d]) for d in axes))) for name, axes in dims.items()}, coords=coords).assign_coords(year=float(year))
        if self.source_members is not None:
            state = state.assign_coords(source_config=("config", list(self.source_members)))
        for name in STATE_NAMES:
            state[name].attrs["units"] = "Mha" if name.startswith("rf_A") else "PgC"
        return state

    def validate(self, state, time):
        # Same checks as comparing against empty_state, without allocating it.
        dims, coordinates = self._validation_template
        for name, expected_dims in dims.items():
            if name not in state or state[name].dims != expected_dims:
                raise ValueError("invalid native state dimensions: " + name)
            values = state[name].values
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("nonfinite or negative state: " + name)
        for name, expected in coordinates.items():
            if not np.array_equal(state[name].values, expected):
                raise ValueError("state coordinate mismatch: " + name)
        if self.source_members is not None and ("source_config" not in state.coords
                or not np.array_equal(state.source_config, self.source_members)):
            raise ValueError("original source_config identities changed")
        future = np.asarray(self.births) > time
        if np.any(state.rf_A_forest.values[..., future] != 0):
            raise ValueError("unborn cohorts cannot have area")
        for area, carbon in (("rf_A_primary", ("rf_W_primary", "rf_F_primary")),
                             ("rf_A_forest", ("rf_W_forest", "rf_F_forest")),
                             ("rf_A_nonforest", ("rf_C_nonforest",))):
            mask = state[area].values == 0
            for name in carbon:
                if np.any(state[name].values[mask] != 0):
                    raise ValueError("zero-area source contains carbon")


def total_carbon(state):
    """Actual system carbon, including products; reference and permafrost excluded."""
    total = None
    for name in STATE_NAMES:
        if name.startswith("rf_A"):
            continue
        values = state[name].values
        part = values.reshape(values.shape[0], -1).sum(axis=1)
        total = part if total is None else total + part
    return xr.DataArray(total, dims=("config",), coords={"config": state.config})


def _flat_sources(state, layout, pwood):
    m, r = len(layout.members), len(layout.regions)
    area = np.concatenate((state.rf_A_primary.values[0, :, None], state.rf_A_forest.values[0].reshape(r, -1), state.rf_A_nonforest.values[0]), axis=1)
    mapping = {c.class_id: layout.biomes.index(c.biome) for c in layout.classes}
    nfbiome = [mapping[c] for c in layout.nonforest_classes]
    c = state.rf_C_nonforest.values
    wood = np.concatenate((state.rf_W_primary.values[:, :, None], state.rf_W_forest.values.reshape(m, r, -1), c * pwood[:, :, nfbiome]), axis=2)
    other = np.concatenate((state.rf_F_primary.values[:, :, None], state.rf_F_forest.values.reshape(m, r, -1), c * (1-pwood[:, :, nfbiome])), axis=2)
    return area, wood, other


def _store_flat(state, layout, area, wood, other):
    m, r, f, k = len(layout.members), len(layout.regions), len(layout.forest_classes), len(layout.births)
    state.rf_A_primary.values[0] = area[:, 0]
    state.rf_W_primary.values[:] = wood[:, :, 0]
    state.rf_F_primary.values[:] = other[:, :, 0]
    state.rf_A_forest.values[0] = area[:, 1:1+f*k].reshape(r, f, k)
    state.rf_W_forest.values[:] = wood[:, :, 1:1+f*k].reshape(m, r, f, k)
    state.rf_F_forest.values[:] = other[:, :, 1:1+f*k].reshape(m, r, f, k)
    state.rf_A_nonforest.values[0] = area[:, 1+f*k:]
    state.rf_C_nonforest.values[:] = wood[:, :, 1+f*k:] + other[:, :, 1+f*k:]


def validate_purpose_rank(layout, purpose_rank, hwp_target_events=None):
    """Validate the narrow fixed-area policy and return its shared source."""
    if purpose_rank is None:
        return None
    if hwp_target_events is not None:
        raise ValueError("purpose_rank cannot be combined with an HWP target")
    if (not isinstance(purpose_rank, Mapping) or len(purpose_rank) != 2
            or set(purpose_rank.values()) != {"youngest", "oldest"}):
        raise ValueError("purpose_rank must map two event IDs to youngest/oldest")
    events = {event.event_id: event for event in layout.events}
    if any(name not in events for name in purpose_rank):
        raise ValueError("purpose_rank event identity missing from layout")
    managed = [events[name] for name in purpose_rank]
    source = managed[0].source_class
    if source not in layout.forest_classes or any(
            event.source_class != source or event.target_class != source
            or event.kind != "harvest" or event.vegetation_action != "clear"
            or event.age_action != "reset" for event in managed):
        raise ValueError("purpose_rank events must clear/reset one shared cohort source")
    if any(event.kind == "harvest" and event.source_class == source
           and event.event_id not in purpose_rank for event in layout.events):
        raise ValueError("third harvest activity competes for purpose_rank source")
    return source


def apply_annual_events(state, layout, time, event_area, weights, pwood, phwp, *, purpose_rank=None, selection_only=False):
    """Select annual activities once and route the same actual source stocks.

    weights have (region,event,source_slot) shape. Event areas are annual Mha.
    Conversions precede harvest, but newly received land cannot be reused.
    Mortality and products from the event are stocks, not continuous inputs.
    purpose_rank explicitly maps two cohort harvest IDs to youngest/oldest.
    Their positive weights must be uniform within each row; zeros retain hard
    eligibility. All carbon routing remains shared with weighted selection.
    """
    if len(layout.members) != 1:
        raise ValueError("Annual event kernel requires one member; native update batches members")
    layout.validate(state, time)
    ranked_source = validate_purpose_rank(layout, purpose_rank)
    r, e, b, p = len(layout.regions), len(layout.events), len(layout.biomes), len(layout.products)
    m, slots = len(layout.members), layout.slots
    s = len(slots)
    event_area, weights, pwood, phwp = map(lambda x: np.asarray(x, dtype=float), (event_area, weights, pwood, phwp))
    for value, shape in ((event_area, (r, e)), (weights, (r, e, s)), (pwood, (m, r, b)), (phwp, (m, r, b, p))):
        if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError("invalid annual event input shape or value")
    if np.any(pwood > 1) or np.any(phwp.sum(axis=-1) > 1):
        raise ValueError("routing fractions exceed one")
    if time not in layout.births and np.any(event_area > 0):
        raise ValueError("annual event time has no reserved birth coordinate")
    classes = {c.class_id: c for c in layout.classes}
    biome = {c.class_id: layout.biomes.index(c.biome) for c in layout.classes}
    area, wood, other = _flat_sources(state, layout, pwood)
    available = area.copy()
    selected = np.zeros((r, e, s))
    for kind, source, rows, cols in layout.event_index_groups:
        if kind == "harvest" and source == ranked_source:
            for ri in range(r):
                d = event_area[ri, rows]
                if not np.any(d):
                    continue
                w = weights[ri][np.ix_(rows, cols)]
                for row_weights in w:
                    positive = row_weights[row_weights > 0]
                    if positive.size and np.any(positive != positive[0]):
                        raise ValueError("purpose_rank cannot combine nonuniform preference weights")
                ages = np.maximum(time - np.array([slots[i][1] for i in cols]), 0.)
                h = select_ranked_area(available[ri, cols], ages, w > 0, d,
                                       [purpose_rank[layout.events[i].event_id] for i in rows])
                selected[ri][np.ix_(rows, cols)] = h
                available[ri, cols] -= h.sum(axis=0)
            continue
        if classes[source].role != "forest_cohort":
            # An unaged source has one slot; no invented age enters w.
            w = (np.asarray(weights[:, rows][:, :, cols]) > 0).astype(float)
        else:
            w = np.asarray(weights[:, rows][:, :, cols])
        d = event_area[:, rows]
        if not np.any(d):
            continue
        h = select_joint_block(available[:, cols], w, d)
        for ri in range(r):
            if not np.any(d[ri]):
                continue
            selected[ri][np.ix_(rows, cols)] = h[ri]
            available[ri, cols] -= h[ri].sum(axis=0)
    tolerance = 1e-10 * max(1., float(area.sum()))
    if np.any(available < -tolerance):
        raise ValueError("annual source area exceeded")
    available = np.maximum(available, 0.)
    # The secondary-harvest preview only needs opening stocks and this selection.
    if selection_only:
        return area, wood, other, selected
    # Partition once: every event and the remaining parcel use the same donor.
    denominator = available + selected.sum(axis=1)
    fraction = np.divide(selected, denominator[:, None, :], out=np.zeros_like(selected), where=denominator[:, None, :] > 0)
    remaining = np.divide(available, denominator, out=np.zeros_like(area), where=denominator > 0)
    new_area = available.copy()
    new_wood, new_other = wood * remaining[None, :, :], other * remaining[None, :, :]
    out = state.copy(deep=True)
    # Most activities use a single current source class and only a small
    # fraction of reserved birth slots. Do not multiply its structural zeros.
    taken_w, taken_f = np.zeros((m, r, e)), np.zeros((m, r, e))
    for ei in range(e):
        columns = np.flatnonzero(np.any(selected[:, ei] > 0, axis=0))
        if columns.size:
            part = fraction[:, ei, columns]
            taken_w[:, :, ei] = np.einsum("rs,mrs->mr", part, wood[..., columns])
            taken_f[:, :, ei] = np.einsum("rs,mrs->mr", part, other[..., columns])
    hwp_added = np.zeros((m, r, e, p))
    cwd_added, soil_added = np.zeros((m, r, e)), np.zeros((m, r, e))
    # Partition old biome dead stocks once, alongside the source parcels.
    # Repeated event-by-event subtraction can leave a negative roundoff tail
    # when the last square hectare leaves a biome.
    area_transfer = np.zeros((r, b, b))
    for si, (c, _) in enumerate(slots):
        area_transfer[:, biome[c], biome[c]] += available[:, si]
    out.rf_C_cwd.values[:] = 0.
    out.rf_C_soil.values[:] = 0.
    for ei, ev in enumerate(layout.events):
        amount = selected[:, ei, :].sum(axis=1)
        if not np.any(amount):
            continue
        if ev.vegetation_action == "unresolved":
            raise ValueError("nonzero event has unresolved vegetation semantics: " + ev.event_id)
        sb, tb = biome[ev.source_class], biome[ev.target_class]
        area_transfer[:, sb, tb] += amount
        if ev.kind == "harvest" and (ev.vegetation_action != "clear" or sb != tb):
            raise ValueError("harvest must clear within its biome")
        retained = ev.vegetation_action == "retain"
        if not retained:
            hwp_added[:, :, ei] = taken_w[:, :, ei, None] * phwp[:, :, sb, :]
            cwd_added[:, :, ei] = taken_w[:, :, ei] - hwp_added[:, :, ei].sum(axis=-1)
            soil_added[:, :, ei] = taken_f[:, :, ei]
            out.rf_C_hwp.values[:, :, sb, :] += hwp_added[:, :, ei]
            out.rf_C_cwd.values[:, :, tb] += cwd_added[:, :, ei]
            out.rf_C_soil.values[:, :, tb] += soil_added[:, :, ei]
        target = classes[ev.target_class]
        if secondary_stand_harvest(ev):
            # Carbon already left with the clear-cut fraction. Put the area back.
            for si in np.flatnonzero(np.any(selected[:, ei] > 0, axis=0)):
                new_area[:, si] += selected[:, ei, si]
        elif ev.age_action != "keep":
            target_birth = time if target.role == "forest_cohort" else None
            ti = slots.index((ev.target_class, target_birth))
            new_area[:, ti] += amount
            if retained:
                new_wood[:, :, ti] += taken_w[:, :, ei]
                new_other[:, :, ti] += taken_f[:, :, ei]
        else:
            # Retained forest relabeling alone needs each original birth slot.
            for si in np.flatnonzero(np.any(selected[:, ei] > 0, axis=0)):
                ti = slots.index((ev.target_class, slots[si][1]))
                new_area[:, ti] += selected[:, ei, si]
                new_wood[:, :, ti] += wood[:, :, si] * fraction[:, ei, si][None, :]
                new_other[:, :, ti] += other[:, :, si] * fraction[:, ei, si][None, :]
    area_partition = area_transfer.sum(axis=-1, keepdims=True)
    fraction_transfer = np.divide(area_transfer, area_partition,
                                  out=np.broadcast_to(np.eye(b), (r, b, b)).copy(),
                                  where=area_partition > 0)
    for name in ("rf_C_cwd", "rf_C_soil"):
        out[name].values[:] += np.einsum("mrs,rst->mrt", state[name].values, fraction_transfer)
    _store_flat(out, layout, new_area, new_wood, new_other)
    out = out.assign_coords(year=float(time))
    residual = (total_carbon(out) - total_carbon(state)).values
    if np.any(abs(residual) > 1e-10 * np.maximum(1., total_carbon(state).values)):
        raise ValueError("annual event carbon budget failed")
    layout.validate(out, time)
    ledger = xr.Dataset({
        "requested_area": (("reg_land", "event"), event_area),
        "executed_area": (("reg_land", "event"), selected.sum(axis=-1)),
        "selected_area": (("reg_land", "event", "source_slot"), selected),
        "source_area": (("reg_land", "source_slot"), area),
        "source_wood": (("config", "reg_land", "source_slot"), wood),
        "source_other": (("config", "reg_land", "source_slot"), other),
        "removed_wood": (("config", "reg_land", "event"), taken_w),
        "removed_other": (("config", "reg_land", "event"), taken_f),
        "hwp_added": (("config", "reg_land", "event", "box_hwp"), hwp_added),
        "cwd_added": (("config", "reg_land", "event"), cwd_added),
        "soil_added": (("config", "reg_land", "event"), soil_added),
        "budget_residual": ("config", residual),
    }, coords=dict(config=list(layout.members), reg_land=list(layout.regions), event=[ev.event_id for ev in layout.events], box_hwp=list(layout.products), source_slot=np.arange(s)))
    ledger["source_birth_time"] = ("source_slot", [np.nan if birth is None else birth for _, birth in slots])
    ledger["source_age"] = time - ledger.source_birth_time
    ledger["source_class"] = ("source_slot", [c for c, _ in slots])
    summaries = event_age_summary(selected, fraction, wood, other, layout, time, phwp)
    ledger = ledger.assign_coords(age_support=list(AGE_SUPPORT), land_action=list(LAND_ACTION))
    for name, value in zip(("area_by_age", "removed_W_by_age", "removed_F_by_age", "HWP_by_age"), summaries):
        dims = (("reg_land", "land_action", "age_support") if name == "area_by_age" else
                ("config", "reg_land", "land_action", "age_support"))
        if name == "HWP_by_age":
            dims += ("box_hwp",)
        ledger[name] = (dims, value)
    for name in ledger.data_vars:
        if name == "source_class":
            continue
        ledger[name].attrs["units"] = "yr" if name in ("source_birth_time", "source_age") else "Mha" if "area" in name else "PgC"
    return out, ledger


@dataclass(frozen=True)
class RegionalRates:
    """Prepared per-member rates; live channels are W, F, then nonforest biomes."""
    production: np.ndarray
    mortality: np.ndarray
    fire: np.ndarray
    crop: np.ndarray
    grazing: np.ndarray
    cwd_fraction: np.ndarray
    dead: tuple


def advance_carbon(state, layout, t0, t1, rates, age_factors=None):
    """Advance stocks and all transfers with frozen environmental rates.

    Live linear integration is exact for frozen production/rates. Live litter
    is supplied uniformly to the dead cascade over the same interval, a
    second-order coupling approximation. Birth times and areas never change.
    """
    dt = float(t1 - t0)
    if dt <= 0:
        raise ValueError("positive integration interval required")
    layout.validate(state, t0)
    m, r, f, k, b = len(layout.members), len(layout.regions), len(layout.forest_classes), len(layout.births), len(layout.biomes)
    shape = (m, r, b+1)
    for name in ("production", "mortality", "fire", "crop", "grazing", "cwd_fraction"):
        value = np.asarray(getattr(rates, name))
        if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError("invalid rate array: " + name)
    if np.any(rates.cwd_fraction > 1) or np.any(rates.crop + rates.grazing > 1):
        raise ValueError("invalid carbon routing fraction")
    if np.any(rates.crop[..., :2] + rates.grazing[..., :2] != 0) or np.any(rates.cwd_fraction[..., 0] != 1) or np.any(rates.cwd_fraction[..., 1] != 0):
        raise ValueError("forest channels require W-to-CWD and F-to-soil mortality")
    factors = np.ones((m, r, 2, f, k)) if age_factors is None else np.asarray(age_factors)
    if factors.shape != (m, r, 2, f, k) or np.any(factors < 0) or not np.all(np.isfinite(factors)):
        raise ValueError("invalid cohort production factors")
    out = state.copy(deep=True)
    ledger = {name: np.zeros((m, r, b)) for name in ("production", "fire", "crop_harvest", "grazing", "live_to_cwd", "live_to_soil")}
    forest_integral = np.zeros((m, r, 2))
    forest_production = np.zeros((m, r, 2))
    age_production = np.zeros((m, r, len(AGE_SUPPORT)))
    support = age_support_index(layout.births, (t0+t1)/2)
    # The coordinates remain complete; only the temporary numerical work is sparse.
    ri, fi, ki = np.nonzero(np.any(state.rf_A_forest.values > 0, axis=0))
    for pool, name in enumerate(("W", "F")):
        stock = state["rf_" + name + "_primary"].values
        production = rates.production[:, :, pool] * state.rf_A_primary.values
        step = advance_live_carbon(stock, production, rates.mortality[:, :, pool], rates.fire[:, :, pool], dt)
        out["rf_" + name + "_primary"].values[:] = step.carbon
        forest_integral[:, :, pool] += step.integrated_carbon
        forest_production[:, :, pool] += step.production
        age_production[:, :, 3] += step.production
        for label, value in (("production", step.production), ("fire", step.fire),
                             ("live_to_cwd" if pool == 0 else "live_to_soil", step.mortality)):
            ledger[label][:, :, 0] += value
        if ri.size:
            stock = state["rf_" + name + "_forest"].values[:, ri, fi, ki]
            production = (rates.production[:, ri, pool] * state.rf_A_forest.values[:, ri, fi, ki]
                          * factors[:, ri, pool, fi, ki])
            step = advance_live_carbon(stock, production, rates.mortality[:, ri, pool], rates.fire[:, ri, pool], dt)
            out["rf_" + name + "_forest"].values[:, ri, fi, ki] = step.carbon
            np.add.at(forest_integral[:, :, pool], (slice(None), ri), step.integrated_carbon)
            np.add.at(forest_production[:, :, pool], (slice(None), ri), step.production)
            np.add.at(age_production, (slice(None), ri, support[ki]), step.production)
            for label, value in (("production", step.production), ("fire", step.fire),
                                 ("live_to_cwd" if pool == 0 else "live_to_soil", step.mortality)):
                np.add.at(ledger[label][:, :, 0], (slice(None), ri), value)
    class_biome = {c.class_id: layout.biomes.index(c.biome) for c in layout.classes}
    for j, name in enumerate(layout.nonforest_classes):
        bi = class_biome[name]
        ch = bi + 1
        step = advance_live_carbon(state.rf_C_nonforest.values[:, :, j], rates.production[:, :, ch] * state.rf_A_nonforest.values[:, :, j], rates.mortality[:, :, ch], rates.fire[:, :, ch], dt, p_charv=rates.crop[:, :, ch], p_graz=rates.grazing[:, :, ch])
        out.rf_C_nonforest.values[:, :, j] = step.carbon
        age_production[:, :, 4] += step.production
        for label in ("production", "fire", "crop_harvest", "grazing"):
            ledger[label][:, :, bi] += getattr(step, label)
        ledger["live_to_cwd"][:, :, bi] += step.mortality * rates.cwd_fraction[:, :, ch]
        ledger["live_to_soil"][:, :, bi] += step.mortality * (1-rates.cwd_fraction[:, :, ch])
    for label in ("cwd_respiration", "soil_respiration", "hwp_respiration"):
        ledger[label] = np.zeros((m, r, b))
    for ri in range(r):
        dead = advance_dead_carbon(state.rf_C_cwd.values[:, ri], state.rf_C_soil.values[:, ri], state.rf_C_hwp.values[:, ri], ledger["live_to_cwd"][:, ri] / dt, ledger["live_to_soil"][:, ri] / dt, np.zeros_like(state.rf_C_hwp.values[:, ri]), rates.dead[ri], dt, member_ids=layout.members, biome_ids=layout.biomes, product_ids=layout.products)
        out.rf_C_cwd.values[:, ri] = dead.cwd
        out.rf_C_soil.values[:, ri] = dead.soil
        out.rf_C_hwp.values[:, ri] = dead.hwp
        ledger["cwd_respiration"][:, ri] = dead.cwd_respiration
        ledger["soil_respiration"][:, ri] = dead.soil_respiration
        ledger["hwp_respiration"][:, ri] = dead.hwp_respiration.sum(axis=-1)
    ledger["net_uptake"] = ledger["production"] - sum(ledger[name] for name in ("fire", "crop_harvest", "grazing", "cwd_respiration", "soil_respiration", "hwp_respiration"))
    out = out.assign_coords(year=float(t1))
    residual = total_carbon(out).values - total_carbon(state).values - ledger["net_uptake"].sum(axis=(1, 2))
    if np.any(abs(residual) > 1e-10 * np.maximum(1., total_carbon(state).values)):
        raise ValueError("continuous carbon budget failed")
    result = xr.Dataset({name: (("config", "reg_land", "bio_land"), value) for name, value in ledger.items()}, coords=dict(config=list(layout.members), reg_land=list(layout.regions), bio_land=list(layout.biomes)))
    result["budget_residual"] = ("config", residual)
    for name in result:
        result[name].attrs["units"] = "PgC"
    for pool, index in (("wood", 0), ("other", 1)):
        result["forest_" + pool + "_production"] = (("config", "reg_land"), forest_production[:, :, index])
        result["forest_" + pool + "_production"].attrs["units"] = "PgC"
        result["forest_" + pool + "_time_integral"] = (("config", "reg_land"), forest_integral[:, :, index])
        result["forest_" + pool + "_time_integral"].attrs["units"] = "PgC yr"
    result = result.assign_coords(age_support=list(AGE_SUPPORT))
    result["production_by_age"] = (("config", "reg_land", "age_support"), age_production)
    result.production_by_age.attrs["units"] = "PgC"
    return out, result

"""Conservative forest area and live-carbon ledger for one region.

Units: area in Mha, total live carbon in Pg C, and time in years. Input
adapters own conversions. Primary forest is unaged; secondary age is
time - birth_time. Births occur at the actual event substep time. Cohorts
are never merged or advected, so aging has no numerical diffusion or tail
outlet; event timing and later carbon integration still need convergence.

Removal is a transfer out of live vegetation, not an atmospheric emission.
The caller must route its carbon to residue, soil and product pools.
Growth, climate responses and annual event scheduling are not implemented.

Modified by the OSCAR-Forest-Age project, 2026-11-04.
Distributed under the repository's CeCILL 2.1 license.
"""

from dataclasses import dataclass, replace
from math import fsum

import numpy as np


def _scalar(value, name, nonnegative=True):
    value = float(value)
    if not np.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def _vector(value, name, nonnegative=True):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if nonnegative and np.any(value < 0):
        raise ValueError(f"{name} must be nonnegative")
    value.setflags(write=False)
    return value


def _check_stocks(area, carbon):
    if area.shape != carbon.shape:
        raise ValueError("area and carbon must have the same shape")
    if np.any((area == 0) & (carbon != 0)):
        raise ValueError("zero area cannot contain live carbon")


@dataclass(frozen=True)
class ForestState:
    """Physical stocks; secondary vectors share the same cohort order."""

    time: float
    primary_area: float
    primary_carbon: float
    birth_time: np.ndarray
    secondary_area: np.ndarray
    secondary_carbon: np.ndarray

    def __post_init__(self):
        for name in ("time", "primary_area", "primary_carbon"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, name != "time"))
        for name in ("birth_time", "secondary_area", "secondary_carbon"):
            object.__setattr__(self, name, _vector(getattr(self, name), name, name != "birth_time"))
        _check_stocks(self.secondary_area, self.secondary_carbon)
        if self.birth_time.shape != self.secondary_area.shape:
            raise ValueError("birth_time must match the secondary stock vectors")
        if np.any(self.birth_time > self.time):
            raise ValueError("secondary cohorts cannot be born in the future")
        if self.primary_area == 0 and self.primary_carbon != 0:
            raise ValueError("zero primary area cannot contain live carbon")

    @property
    def age(self):
        """Secondary ages in years; primary forest has no finite age."""
        return self.time - self.birth_time

    @property
    def total_area(self):
        return fsum((self.primary_area, *self.secondary_area))

    @property
    def total_carbon(self):
        return fsum((self.primary_carbon, *self.secondary_carbon))


@dataclass(frozen=True)
class Removal:
    """Transferred stocks, with primary first and then source cohort order."""

    area: np.ndarray
    carbon: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "area", _vector(self.area, "removed area"))
        object.__setattr__(self, "carbon", _vector(self.carbon, "removed carbon"))
        _check_stocks(self.area, self.carbon)

    @property
    def total_area(self):
        return fsum(self.area)

    @property
    def total_carbon(self):
        return fsum(self.carbon)


def select_area(area, birth_time, demand, strategy="proportional"):
    """Select one feasible area demand; carbon never affects this choice.

    'oldest' selects earliest births first; 'youngest' selects latest births.
    Equal birth times retain source order. Requests above supply are rejected,
    not capped. No age preference is imposed on primary forest.
    """
    area = _vector(area, "area")
    birth_time = _vector(birth_time, "birth_time", nonnegative=False)
    demand = _scalar(demand, "demand")
    if birth_time.shape != area.shape:
        raise ValueError("birth_time must match area")
    if strategy not in ("proportional", "oldest", "youngest"):
        raise ValueError("strategy must be proportional, oldest or youngest")
    supply = fsum(area)
    if demand > supply:
        raise ValueError("area demand exceeds available supply")
    if demand == 0:
        return np.zeros_like(area)
    if demand == supply:
        return area.copy()
    if strategy == "proportional":
        return area * (demand / supply)
    order = np.argsort(birth_time if strategy == "oldest" else -birth_time, kind="stable")
    selected = np.zeros_like(area)
    remaining = demand
    for index in order:
        selected[index] = min(area[index], remaining)
        remaining -= selected[index]
        if remaining == 0:
            break
    return selected


def reserve_area(available, selections):
    """Validate simultaneous reservations and return their cohort-wise sum.

    Rows are events, columns are source cohorts. The scheduler must resolve
    preferences and shortages before calling this function. It neither
    scales demands nor assigns priority. Newly arriving area is unavailable.
    """
    available = _vector(available, "available")
    selections = np.asarray(selections, dtype=np.float64)
    if selections.ndim != 2 or selections.shape[1] != available.size:
        raise ValueError("selections must be an event-by-cohort matrix")
    if not np.all(np.isfinite(selections)) or np.any(selections < 0):
        raise ValueError("selections must be finite and nonnegative")
    reserved = np.array([fsum(column) for column in selections.T], dtype=np.float64)
    if np.any(reserved > available):
        raise ValueError("shared reservations exceed available cohort area")
    return reserved


def remove(state, secondary_area, primary_area=0):
    """Remove selected area and its existing live carbon exactly once.

    Returns (remaining_state, transferred_stocks). Partial removal uses
    C * (selected_area / source_area), algebraically selected_area * C/A.
    Taking the entire source takes its entire carbon, with no residual.
    """
    secondary_area = _vector(secondary_area, "selected secondary area")
    primary_area = _scalar(primary_area, "selected primary area")
    available = np.concatenate(([state.primary_area], state.secondary_area))
    carbon = np.concatenate(([state.primary_carbon], state.secondary_carbon))
    selected = np.concatenate(([primary_area], secondary_area))
    selected = reserve_area(available, selected[None, :])
    fraction = np.divide(selected, available, out=np.zeros_like(selected), where=available > 0)
    moved = Removal(selected, carbon * fraction)
    area_left = available - moved.area
    carbon_left = carbon - moved.carbon
    remaining = replace(
        state, primary_area=area_left[0], primary_carbon=carbon_left[0],
        secondary_area=area_left[1:], secondary_carbon=carbon_left[1:],
    )
    return remaining, moved


def establish(state, area, carbon=0):
    """Add a zero-age secondary cohort at state.time, retaining supplied carbon.

    Supplied stocks must come from a caller's external land-transfer ledger.
    Existing cohorts, including empty cohorts, keep their original indices.
    Different event times are never combined, even within the same year.
    """
    area = _scalar(area, "established area")
    carbon = _scalar(carbon, "established carbon")
    if area == 0:
        if carbon != 0:
            raise ValueError("zero established area cannot contain live carbon")
        return state
    return replace(
        state, birth_time=np.append(state.birth_time, state.time),
        secondary_area=np.append(state.secondary_area, area),
        secondary_carbon=np.append(state.secondary_carbon, carbon),
    )


def harvest(state, secondary_area, primary_area=0):
    """Clear selected stands and reset the same area to zero-age secondary.

    The returned Removal records the cleared source stocks, including
    primary stocks. All removed carbon awaits downstream routing; none is
    silently retained in the new cohort or assigned to the atmosphere.
    """
    remaining, moved = remove(state, secondary_area, primary_area)
    return establish(remaining, moved.total_area), moved


def advance(state, dt):
    """Advance the clock in years without changing area, carbon or births."""
    dt = _scalar(dt, "dt")
    time = state.time + dt
    if not np.isfinite(time) or (dt > 0 and time <= state.time):
        raise ValueError("time increment is not representable")
    return replace(state, time=time)


# Two-pool candidate: common area, vectorized parameter members, exact births.
POOL_NAMES = ("W", "F")


def _member_ids(values):
    """Preserve explicit parameter identities; never infer them from row order."""
    values = tuple(values)
    if not values or any(not isinstance(x, (str, int, np.integer)) for x in values):
        raise ValueError("member_ids must be nonempty string/integer identities")
    if len(set(values)) != len(values):
        raise ValueError("member_ids must be unique")
    return tuple(int(x) if isinstance(x, np.integer) else x for x in values)


def align_members(values, source_ids, target_ids):
    """Align the first array axis by identity; missing/extra members raise."""
    source_ids, target_ids = _member_ids(source_ids), _member_ids(target_ids)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[0] != len(source_ids):
        raise ValueError("first axis must match source member_ids")
    if set(source_ids) != set(target_ids):
        raise ValueError("source and target member_ids differ")
    index = {key: i for i, key in enumerate(source_ids)}
    return values[[index[key] for key in target_ids]].copy()


def _carbon_array(value, shape, name):
    value = np.array(value, dtype=np.float64, copy=True)
    if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must have shape {shape} and finite nonnegative carbon")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class TwoPoolForestState:
    """One region: common areas and (member, pool, cohort) actual PgC stocks.

    W follows the candidate wood definition. F is the remaining effective live
    carbon, not a pure leaf/fine-root or universally fast pool. Empty cohorts
    retain their integer positions; primary forest has no finite age.
    """

    time: float
    member_ids: tuple
    primary_area: float
    primary_carbon: np.ndarray
    birth_time: np.ndarray
    secondary_area: np.ndarray
    secondary_carbon: np.ndarray
    region_id: str
    entry_origin: tuple

    def __post_init__(self):
        object.__setattr__(self, "member_ids", _member_ids(self.member_ids))
        if not isinstance(self.region_id, str) or not self.region_id:
            raise ValueError("region_id must be a nonempty string")
        object.__setattr__(self, "time", _scalar(self.time, "time", nonnegative=False))
        object.__setattr__(self, "primary_area", _scalar(self.primary_area, "primary_area"))
        object.__setattr__(self, "birth_time", _vector(self.birth_time, "birth_time", False))
        object.__setattr__(self, "secondary_area", _vector(self.secondary_area, "secondary_area"))
        m, k = len(self.member_ids), self.secondary_area.size
        from .land_geometry import ENTRY_ORIGINS
        origins = tuple(self.entry_origin)
        if len(origins) != k or any(x not in ENTRY_ORIGINS for x in origins):
            raise ValueError("entry_origin must explicitly match every finite forest cohort")
        object.__setattr__(self, "entry_origin", origins)
        for name, shape in (("primary_carbon", (m, 2)), ("secondary_carbon", (m, 2, k))):
            object.__setattr__(self, name, _carbon_array(getattr(self, name), shape, name))
        if self.birth_time.shape != (k,) or np.any(self.birth_time > self.time):
            raise ValueError("birth_time must match cohorts and not be in the future")
        if self.primary_area == 0 and np.any(self.primary_carbon != 0):
            raise ValueError("zero primary area cannot contain live carbon")
        if np.any(self.secondary_carbon[:, :, self.secondary_area == 0] != 0):
            raise ValueError("zero secondary area cannot contain live carbon")

    @property
    def age(self):
        return self.time - self.birth_time

    @property
    def total_area(self):
        return fsum((self.primary_area, *self.secondary_area))

    @property
    def pool_carbon(self):
        """Actual stocks by member and pool; never sum parameter members."""
        return self.primary_carbon + self.secondary_carbon.sum(axis=2)

    @property
    def total_carbon(self):
        return self.pool_carbon.sum(axis=1)

    def snapshot(self):
        """JSON-compatible forest state, not a full host/run checkpoint."""
        return dict(schema="forest-two-pool-v2", time=self.time, region_id=self.region_id,
                    member_ids=list(self.member_ids), pool=list(POOL_NAMES),
                    cohort_id=list(range(self.secondary_area.size)),
                    primary_area=self.primary_area, primary_carbon=self.primary_carbon.tolist(),
                    birth_time=self.birth_time.tolist(), entry_origin=list(self.entry_origin),
                    secondary_area=self.secondary_area.tolist(), secondary_carbon=self.secondary_carbon.tolist())

    @classmethod
    def from_snapshot(cls, snapshot):
        data = dict(snapshot)
        if data.pop("schema") != "forest-two-pool-v2" or tuple(data.pop("pool")) != POOL_NAMES:
            raise ValueError("unsupported forest schema or pool order")
        if data.pop("cohort_id") != list(range(len(data["birth_time"]))):
            raise ValueError("cohort identity/order changed")
        return cls(**data)


@dataclass(frozen=True)
class TwoPoolRemoval:
    """Removed PgC by member, pool, source (primary first), one area vector."""

    member_ids: tuple
    area: np.ndarray
    carbon: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "member_ids", _member_ids(self.member_ids))
        object.__setattr__(self, "area", _vector(self.area, "removed area"))
        shape = (len(self.member_ids), 2, self.area.size)
        object.__setattr__(self, "carbon", _carbon_array(self.carbon, shape, "removed carbon"))
        if np.any(self.carbon[:, :, self.area == 0] != 0):
            raise ValueError("zero removed area cannot contain carbon")

    @property
    def total_area(self):
        return fsum(self.area)

    @property
    def total_carbon(self):
        return self.carbon.sum(axis=(1, 2))


def remove_two_pool(state, secondary_area, primary_area=0):
    """One common selection removes actual W and F, with no mature re-split."""
    selected = np.concatenate(([_scalar(primary_area, "selected primary area")],
                               _vector(secondary_area, "selected secondary area")))
    available = np.concatenate(([state.primary_area], state.secondary_area))
    selected = reserve_area(available, selected[None, :])
    fraction = np.divide(selected, available, out=np.zeros_like(selected), where=available > 0)
    carbon = np.concatenate((state.primary_carbon[:, :, None], state.secondary_carbon), axis=2)
    moved = TwoPoolRemoval(state.member_ids, selected, carbon * fraction[None, None, :])
    left = carbon - moved.carbon
    area_left = available - selected
    return replace(state, primary_area=area_left[0], secondary_area=area_left[1:],
                   primary_carbon=left[:, :, 0], secondary_carbon=left[:, :, 1:]), moved


def establish_two_pool(state, area, *, entry_origin, retained_carbon=None, member_ids=None):
    """Add a zero-age cohort; retained W/F must be supplied explicitly.

    The caller subtracts the same amounts from its donor. A total-carbon-only
    donor is rejected: this function never infers wood from the target p_wood.
    Omitting retained_carbon denotes zero-carbon establishment.
    """
    from .land_geometry import ENTRY_ORIGINS
    if entry_origin not in ENTRY_ORIGINS:
        raise ValueError("explicit accepted entry_origin is required")
    area = _scalar(area, "established area")
    if retained_carbon is None:
        carbon = np.zeros((len(state.member_ids), 2))
    else:
        if member_ids is None:
            raise ValueError("explicit member_ids required for retained carbon")
        carbon = align_members(retained_carbon, member_ids, state.member_ids)
    carbon = _carbon_array(carbon, (len(state.member_ids), 2), "retained W/F carbon")
    if entry_origin == "cleared" and np.any(carbon != 0):
        raise ValueError("cleared establishment must have zero retained live carbon")
    if area == 0:
        if np.any(carbon != 0):
            raise ValueError("zero established area cannot contain carbon")
        return state
    return replace(state, birth_time=np.append(state.birth_time, state.time),
                   entry_origin=state.entry_origin + (entry_origin,),
                   secondary_area=np.append(state.secondary_area, area),
                   secondary_carbon=np.concatenate((state.secondary_carbon, carbon[:, :, None]), axis=2))


def harvest_two_pool(state, secondary_area, primary_area=0):
    remaining, moved = remove_two_pool(state, secondary_area, primary_area)
    return establish_two_pool(remaining, moved.total_area, entry_origin="cleared"), moved

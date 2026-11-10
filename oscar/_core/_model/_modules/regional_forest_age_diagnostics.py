"""Small age-support summaries of the existing area/carbon transactions.

These labels describe current age or event-source age, never cohort history.
The last class keeps nonforest activity visible rather than assigning it age.
"""
import numpy as np
import xarray as xr
_EXCHANGE_CACHE = {}


def cached_dataarray(key, build):
    """Reuse one zero exchange shell. The returned array has its own data."""
    value = _EXCHANGE_CACHE.get(key)
    if value is None:
        value = build()
        _EXCHANGE_CACHE[key] = value
    return value.copy(deep=True)



AGE_SUPPORT = ("0_to_5", "5_to_100", "over_100", "primary_unaged", "nonforest")
LAND_ACTION = ("harvest", "conversion_clear", "conversion_retain")
AGE_EXCHANGES = ("rf_age_area_mean", "rf_age_NPP", "rf_event_area_by_age",
                 "rf_event_removed_W_by_age", "rf_event_removed_F_by_age", "rf_event_HWP_by_age")


def birth_support(ages):
    """Map cohort ages onto [0,5), [5,100], (100, infinity)."""
    ages = np.asarray(ages, dtype=float)
    return np.where(ages < 5., 0, np.where(ages <= 100., 1, 2))


def age_support_index(births, time):
    """Disjoint support: [0,5), [5,100], (100,infinity).

    Future slots also map to the first class but must carry zero area/carbon.
    Their numerical storage does not create any physical contribution.
    """
    return birth_support(time - np.asarray(births))


def area_by_age(state, layout, time, cohort_ages=None):
    result = np.zeros((len(layout.members), len(layout.regions), len(AGE_SUPPORT)))
    support = age_support_index(layout.births, time) if cohort_ages is None else birth_support(cohort_ages)
    for index in range(3):
        result[:, :, index] = state.rf_A_forest.values[..., support == index].sum(axis=(2, 3))
    result[:, :, 3] = state.rf_A_primary.values
    result[:, :, 4] = state.rf_A_nonforest.values.sum(axis=2)
    return xr.DataArray(result, dims=("config", "reg_land", "age_support"),
                        coords=dict(config=list(layout.members), reg_land=list(layout.regions), age_support=list(AGE_SUPPORT)))


def event_age_summary(selected, fraction, wood, other, layout, time, phwp, cohort_ages=None):
    """Summarize the SAME selected parcels and source-stock fractions.

    No activity is selected again. Retain transfers are distinct from clearance;
    their source carbon remains live at the destination and generates no HWP.
    """
    m, r = wood.shape[:2]
    area = np.zeros((r, len(LAND_ACTION), len(AGE_SUPPORT)))
    carbon = [np.zeros((m, r, len(LAND_ACTION), len(AGE_SUPPORT))) for _ in range(2)]
    product = np.zeros(carbon[0].shape + (len(layout.products),))
    bins = age_support_index(layout.births, time) if cohort_ages is None else birth_support(cohort_ages)
    source_bin = np.r_[3, np.tile(bins, len(layout.forest_classes)),
                       np.full(len(layout.nonforest_classes), 4)]
    mapping = {c.class_id: layout.biomes.index(c.biome) for c in layout.classes}
    for ei, event in enumerate(layout.events):
        used = np.flatnonzero(np.any(selected[:, ei] > 0, axis=0))
        if not used.size:
            continue
        action = (0 if event.kind == "harvest" else
                  1 if event.vegetation_action == "clear" else 2)
        product_share = phwp[:, :, mapping[event.source_class]]
        for index in range(len(AGE_SUPPORT)):
            columns = used[source_bin[used] == index]
            if not columns.size:
                continue
            area[:, action, index] += selected[:, ei, columns].sum(axis=-1)
            removed_w = (fraction[:, ei, columns][None, ...] * wood[..., columns]).sum(axis=-1)
            removed_f = (fraction[:, ei, columns][None, ...] * other[..., columns]).sum(axis=-1)
            carbon[0][:, :, action, index] += removed_w
            carbon[1][:, :, action, index] += removed_f
            if event.vegetation_action == "clear":
                product[:, :, action, index] += removed_w[..., None] * product_share
    return area, *carbon, product


def initial_age_exchange(name, state, layout, time):
    """Initial area is a state; integrated production/activity starts at zero."""
    if name == "rf_age_area_mean":
        return area_by_age(state, layout, time)
    def build():
        coordinates = dict(config=list(layout.members), reg_land=list(layout.regions),
                           land_action=list(LAND_ACTION), age_support=list(AGE_SUPPORT), box_hwp=list(layout.products))
        dims = (("config", "reg_land", "age_support") if name == "rf_age_NPP" else
                ("config", "reg_land", "land_action", "age_support"))
        if name == "rf_event_HWP_by_age":
            dims += ("box_hwp",)
        return xr.DataArray(np.zeros(tuple(len(coordinates[d]) for d in dims)), dims=dims,
                            coords={d: coordinates[d] for d in dims})
    return cached_dataarray(("age", name, tuple(layout.members), tuple(layout.regions), tuple(layout.products)), build)


HARVEST_EXCHANGES = ('rf_harvest_area', 'rf_harvest_age_moment', 'rf_harvest_area_by_age',
                     'rf_harvest_removed_W', 'rf_harvest_removed_F', 'rf_harvest_HWP')
HARVEST_UNITS = {'rf_harvest_area': 'Mha yr-1', 'rf_harvest_age_moment': 'Mha yr yr-1',
                 'rf_harvest_area_by_age': 'Mha yr-1', 'rf_harvest_removed_W': 'PgC yr-1',
                 'rf_harvest_removed_F': 'PgC yr-1', 'rf_harvest_HWP': 'PgC yr-1'}


def initial_harvest_exchange(name, layout):
    """Zero rates at initialization/non-event steps; source labels are explicit."""
    def build():
        events = tuple(event for event in layout.events if event.kind == 'harvest')
        coordinates = dict(config=list(layout.members), reg_land=list(layout.regions),
                           harvest_event=[event.event_id for event in events],
                           age_support=list(AGE_SUPPORT), box_hwp=list(layout.products))
        dims = ('config', 'reg_land', 'harvest_event')
        if name == 'rf_harvest_area_by_age':
            dims += ('age_support',)
        if name == 'rf_harvest_HWP':
            dims += ('box_hwp',)
        return xr.DataArray(np.zeros(tuple(len(coordinates[d]) for d in dims)), dims=dims,
                            coords={d: coordinates[d] for d in dims}, attrs={'units': HARVEST_UNITS[name]})
    return cached_dataarray(("harvest", name, tuple(layout.members), tuple(layout.regions),
                         tuple(event.event_id for event in layout.events if event.kind == "harvest"),
                         tuple(layout.products)), build)


def _event_values(ledger, name, index):
    """Harvest rows of one ledger variable, with config on axis 0."""
    variable = ledger.variables[name]
    values = np.asarray(variable.data)
    dims = tuple(variable.dims)
    if "config" not in dims:
        values = np.expand_dims(values, 0)
        dims = ("config",) + dims
    order = ("config", "reg_land", "event") + tuple(dim for dim in dims if dim not in ("config", "reg_land", "event"))
    values = np.transpose(values, [dims.index(dim) for dim in order])
    return np.take(values, index, axis=2), order


def harvest_source_summary(ledger, layout):
    """Reduce existing actual transfers; do not select or route carbon again.

    Returned amounts are per annual event. The native caller divides by its
    event substep duration; the host then forms duration-weighted interval means.
    The age moment is selected finite-age area times actual source age, Mha yr.
    Primary and nonforest parcels never receive an invented finite age.
    """
    index = [i for i, event in enumerate(layout.events) if event.kind == "harvest"]
    events = [layout.events[i].event_id for i in index]
    selected, _ = _event_values(ledger, "selected_area", index)
    ages = np.asarray(ledger.variables["source_age"].data, dtype=float)
    finite = np.isfinite(ages)
    moment = (selected * np.where(finite, ages, 0.0)).sum(axis=-1)
    # Source-slot categories follow the existing prepared class roles, not names.
    roles = {item.class_id: item.role for item in layout.classes}
    bins = np.full(ages.shape, 4, dtype=int)
    bins[finite] = np.where(ages[finite] < 5.0, 0, np.where(ages[finite] <= 100.0, 1, 2))
    for slot, source in enumerate(np.asarray(ledger.variables["source_class"].data)):
        if roles[str(source)] == "forest_unaged":
            bins[slot] = 3
    by_age = np.stack([selected[..., bins == band].sum(axis=-1) for band in range(len(AGE_SUPPORT))], axis=-1)
    removed_w, _ = _event_values(ledger, "removed_wood", index)
    removed_f, _ = _event_values(ledger, "removed_other", index)
    hwp, hwp_dims = _event_values(ledger, "hwp_added", index)
    config = list(np.asarray(ledger.config.values).tolist())
    result = xr.Dataset({
        "rf_harvest_area": (("config", "reg_land", "harvest_event"), selected.sum(axis=-1)),
        "rf_harvest_age_moment": (("config", "reg_land", "harvest_event"), moment),
        "rf_harvest_area_by_age": (("config", "reg_land", "harvest_event", "age_support"), by_age),
        "rf_harvest_removed_W": (("config", "reg_land", "harvest_event"), removed_w),
        "rf_harvest_removed_F": (("config", "reg_land", "harvest_event"), removed_f),
        "rf_harvest_HWP": (("config", "reg_land", "harvest_event", "box_hwp"), hwp),
    }, coords={
        "config": config,
        "reg_land": list(np.asarray(ledger.reg_land.values).tolist()),
        "harvest_event": events,
        "age_support": list(AGE_SUPPORT),
        "box_hwp": list(np.asarray(ledger.box_hwp.values).tolist()),
    })
    del hwp_dims
    for name in HARVEST_EXCHANGES:
        result[name].attrs = {'units': ('Mha yr' if name == 'rf_harvest_age_moment' else
                                       'Mha' if name in ('rf_harvest_area', 'rf_harvest_area_by_age') else 'PgC'),
                              'time_support': 'Amount at this annual event'}
    return result

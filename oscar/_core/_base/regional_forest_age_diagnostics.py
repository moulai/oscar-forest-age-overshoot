"""Small age-support summaries of the existing area/carbon transactions.

These labels describe current age or event-source age, never cohort history.
The last class keeps nonforest activity visible rather than assigning it age.
"""
import numpy as np
import xarray as xr
import weakref

_EXCHANGE_CACHE = {}


def cached_dataarray(owner, name, build):
    """Reuse the coordinate shell of a zero exchange. Copy data before return."""
    key = (id(owner), name)
    slot = _EXCHANGE_CACHE.get(key)
    if slot is None or slot[0]() is not owner:
        try:
            ref = weakref.ref(owner)
        except TypeError:
            ref = lambda owner=owner: owner
        slot = (ref, build())
        _EXCHANGE_CACHE[key] = slot
    return slot[1].copy(deep=True)



AGE_SUPPORT = ("0_to_5", "5_to_100", "over_100", "primary_unaged", "nonforest")
LAND_ACTION = ("harvest", "conversion_clear", "conversion_retain")
AGE_EXCHANGES = ("rf_age_area_mean", "rf_age_NPP", "rf_event_area_by_age",
                 "rf_event_removed_W_by_age", "rf_event_removed_F_by_age", "rf_event_HWP_by_age")


def age_support_index(births, time):
    """Disjoint support: [0,5), [5,100], (100,infinity).

    Future slots also map to the first class but must carry zero area/carbon.
    Their numerical storage does not create any physical contribution.
    """
    age = time - np.asarray(births)
    return np.where(age < 5., 0, np.where(age <= 100., 1, 2))


def area_by_age(state, layout, time):
    result = np.zeros((len(layout.members), len(layout.regions), len(AGE_SUPPORT)))
    support = age_support_index(layout.births, time)
    for index in range(3):
        result[:, :, index] = state.rf_A_forest.values[..., support == index].sum(axis=(2, 3))
    result[:, :, 3] = state.rf_A_primary.values
    result[:, :, 4] = state.rf_A_nonforest.values.sum(axis=2)
    return xr.DataArray(result, dims=("config", "reg_land", "age_support"),
                        coords=dict(config=list(layout.members), reg_land=list(layout.regions), age_support=list(AGE_SUPPORT)))


def event_age_summary(selected, fraction, wood, other, layout, time, phwp):
    """Summarize the SAME selected parcels and source-stock fractions.

    No activity is selected again. Retain transfers are distinct from clearance;
    their source carbon remains live at the destination and generates no HWP.
    """
    m, r = wood.shape[:2]
    area = np.zeros((r, len(LAND_ACTION), len(AGE_SUPPORT)))
    carbon = [np.zeros((m, r, len(LAND_ACTION), len(AGE_SUPPORT))) for _ in range(2)]
    product = np.zeros(carbon[0].shape + (len(layout.products),))
    source_bin = np.r_[3, np.tile(age_support_index(layout.births, time), len(layout.forest_classes)),
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
    return cached_dataarray(layout, "age:" + name, build)


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
    return cached_dataarray(layout, "harvest:" + name, build)


def harvest_source_summary(ledger, layout):
    """Reduce existing actual transfers; do not select or route carbon again.

    Returned amounts are per annual event. The native caller divides by its
    event substep duration; the host then forms duration-weighted interval means.
    The age moment is selected finite-age area times actual source age, Mha yr.
    Primary and nonforest parcels never receive an invented finite age.
    """
    events = [event.event_id for event in layout.events if event.kind == 'harvest']
    selected = ledger.selected_area.sel(event=events).rename(event='harvest_event')
    # The pure member event kernel retains local parcel matrices. Its public
    # summaries, like native interval diagnostics, always retain config.
    if "config" not in selected.dims:
        selected = selected.expand_dims(config=ledger.config)
    ages = ledger.source_age.values
    finite = np.isfinite(ages)
    moment = (selected * xr.DataArray(np.where(finite, ages, 0.), dims='source_slot')).sum('source_slot')
    # Source-slot categories follow the existing prepared class roles, not names.
    roles = {item.class_id: item.role for item in layout.classes}
    bins = np.full(len(ages), 4, dtype=int)
    bins[finite] = np.where(ages[finite] < 5., 0, np.where(ages[finite] <= 100., 1, 2))
    for index, source in enumerate(ledger.source_class.values):
        if roles[str(source)] == 'forest_unaged':
            bins[index] = 3
    by_age = xr.concat([selected.isel(source_slot=np.flatnonzero(bins == index)).sum('source_slot')
                       for index in range(len(AGE_SUPPORT))],
                      dim=xr.IndexVariable('age_support', list(AGE_SUPPORT))).transpose('config', 'reg_land', 'harvest_event', 'age_support')
    def transferred(name):
        return ledger[name].sel(event=events).rename(event='harvest_event')
    result = xr.Dataset({'rf_harvest_area': selected.sum('source_slot'),
                       'rf_harvest_age_moment': moment,
                       'rf_harvest_area_by_age': by_age,
                       'rf_harvest_removed_W': transferred('removed_wood'),
                       'rf_harvest_removed_F': transferred('removed_other'),
                       'rf_harvest_HWP': transferred('hwp_added')})
    for name in HARVEST_EXCHANGES:
        result[name].attrs = {'units': ('Mha yr' if name == 'rf_harvest_age_moment' else
                                       'Mha' if name in ('rf_harvest_area', 'rf_harvest_area_by_age') else 'PgC'),
                              'time_support': 'Amount at this annual event'}
    return result

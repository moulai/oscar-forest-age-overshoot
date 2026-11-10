"""Finite secondary-harvest candidate using an operational aboveground projection.

The prepared request is gross carbon allocated to repeated secondary harvest.
It is neither HWP delivery nor a whole-land wood target. W/F state and routing
are unchanged; the projection is a working approximation, not measured organs.
"""
from math import fsum

import numpy as np
import xarray as xr

from .regional_forest_age import apply_annual_events
from .regional_forest_age_diagnostics import cached_dataarray
from .regional_forest_management import _managed_rows
from .regional_forest_age_allocation import select_area

CANDIDATE = 'secondary_gross_agb_request_proportional'
WEIGHTED_CANDIDATE = 'secondary_gross_agb_request_weighted'
PROJECTION = 'native_effective_aboveground'
SEC_UNITS = {'rf_sec_agb_' + name: 'PgC yr-1' for name in
             ('request', 'capacity', 'capacity_soft', 'achieved', 'shortfall', 'removed_soft', 'removed_W',
              'removed_F', 'HWP', 'CWD', 'soil')}
SEC_UNITS.update(rf_sec_agb_area='Mha yr-1', rf_sec_agb_input_area='Mha yr-1')
SEC_EXCHANGES = tuple(SEC_UNITS)
SEC_INPUTS = ('Q_sec_harvest', 'Q_sec_harvest_by_label')


def validate_secondary_contract(layout, events, projection, hwp_events=None,
                                purpose_rank=None, hwp_rule='weighted', secondary_harvest_contract=CANDIDATE):
    if secondary_harvest_contract not in (CANDIDATE, WEIGHTED_CANDIDATE):
        raise ValueError('unknown secondary harvest contract')
    if events is None:
        if secondary_harvest_contract != CANDIDATE:
            raise ValueError('weighted secondary contract requires secondary_agb_events')
        if projection is not None:
            raise ValueError('harvest_projection requires secondary_agb_events')
        return
    _managed_rows(layout, events)
    if projection != PROJECTION:
        raise ValueError('secondary carbon candidate requires native_effective_aboveground')
    if hwp_events is not None or purpose_rank is not None or hwp_rule != 'weighted':
        raise ValueError('secondary carbon candidate cannot combine HWP targets or rank policies')


def initial_secondary_exchange(name, state, events):
    def build():
        value = xr.zeros_like(state.rf_W_primary)
        if name == 'rf_sec_agb_input_area':
            value = value.expand_dims(secondary_harvest_label=list(events), axis=-1)
        return value
    # Primary's labels identify the member/region shell. Carbon values do not.
    return cached_dataarray(state.rf_W_primary.attrs, name if name != 'rf_sec_agb_input_area' else name + ':' + ','.join(events), build)


def select_secondary_area(area, weight, density, target):
    """Select h=A*min(1,lambda*w) for gross carbon, including zero-G parcels.

    Reuse the existing capacity solver on positive-yield parcels. Its selected
    fractions determine lambda; when all positive-G parcels saturate, use the
    left edge of that plateau. Zero weights remain forbidden. No positive
    capacity means lambda=0 and no area is selected, even if Q is positive.
    """
    selected, _ = select_area(area, weight, target, yield_density=density)
    area, weight, density = (np.asarray(x, dtype=float) for x in (area, weight, density))
    positive = (area > 0) & (weight > 0) & (density > 0)
    if target == 0 or not positive.any():
        return np.zeros_like(area)
    eligible = (area > 0) & (weight > 0)
    capacity = float(np.where(eligible, area*density, 0.).sum())
    # Preserve the original uniform contract, including its zero-G area share.
    if np.all(weight[eligible] == weight[eligible][0]):
        fraction = min(1., target / capacity)
        return np.where(eligible, area*fraction, 0.)
    unsaturated = positive & (selected > 0) & (selected < area)
    # Sum the signed residual together; rounded sum(G)==Q need not mean
    # full supply, and the wrong plateau can overselect large zero-G areas.
    if fsum([*(area[positive]*density[positive]), -target]) <= 0:
        fraction, scale = 1., weight[positive].min()
    elif unsaturated.any():
        index = np.flatnonzero(unsaturated)[np.argmax(weight[unsaturated])]
        fraction = selected[index] / area[index]
        scale = weight[index]
    elif np.any(selected > 0):
        # Roundoff can exhaust the target when one source just saturates.
        fraction, scale = 1., weight[selected > 0].min()
    else:
        return selected
    # Ratios avoid constructing a possibly overflowing absolute lambda.
    with np.errstate(over='ignore'):
        fractions = np.minimum(1., (weight / scale) * fraction)
    return area * fractions


def apply_secondary_agb_request_events(state, layout, time, event_area, weights,
                                       pwood, phwp, target_q, label_q, k_f_ag, *,
                                       secondary_events, projection, k_f_ag_leaf_only,
                                       secondary_harvest_contract=CANDIDATE):
    """Resolve one annual secondary request from actual remaining opening stock.

    Q and label_q have (region,) and (region, 2) shape, in PgC per interval.
    k_f_ag has (config, region) shape and is prepared as
    (1-p_wood-p_root)/(1-p_wood). Eligible remaining cohorts share a single
    fraction f=min(1,Q/sum(W+k*F)) under the default contract. The optional
    weighted contract uses min(1,lambda*w), with the smallest feasible lambda;
    label Q fractions only partition the common selected matrix.
    Zero-carbon eligible parcels retain their area share. No incoming parcel
    can supply the event. A pure common preview reserves conversions; the
    final event kernel is applied to the original state exactly once.
    k_f_ag_leaf_only is used only to diagnose soft carbon as (k-k_leaf)*F;
    it does not alter the projection used to meet Q, the state, or selection.
    """
    if len(layout.members) != 1:
        raise ValueError("Secondary event kernel requires one member; native update batches members")
    validate_secondary_contract(layout, secondary_events, projection,
                                secondary_harvest_contract=secondary_harvest_contract)
    rows, source = _managed_rows(layout, secondary_events)
    if any(ev.kind == 'harvest' and ev.source_class == source and i not in rows
           for i, ev in enumerate(layout.events)):
        raise ValueError('other harvest activities compete for the secondary source')
    slots = layout.slots
    r, e, s = len(layout.regions), len(layout.events), len(slots)
    request = np.array(event_area, dtype=float, copy=True)
    weights = np.asarray(weights, dtype=float)
    target, labels, k, k_leaf = (np.asarray(x, dtype=float) for x in
                                (target_q, label_q, k_f_ag, k_f_ag_leaf_only))
    for value, shape in ((request, (r,e)), (weights, (r,e,s)), (target, (r,)),
                         (labels, (r,2)), (k, (1,r)), (k_leaf, (1,r))):
        if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError('invalid secondary request, weights, or projection input')
    if np.any(k > 1) or np.any(k_leaf > k):
        raise ValueError('projection fractions must lie in 0 <= k_F_AG_leaf_only <= k_F_AG <= 1')
    if not np.allclose(labels.sum(axis=-1), target, rtol=1e-12, atol=1e-14):
        raise ValueError('secondary label requests must sum to Q_sec_harvest')
    cols = [i for i,(name,_) in enumerate(slots) if name == source]
    eligible = weights[:, rows[0], cols] > 0
    if not np.array_equal(eligible, weights[:, rows[1], cols] > 0):
        raise ValueError('secondary labels require identical hard eligibility')
    if secondary_harvest_contract == WEIGHTED_CANDIDATE:
        if not np.array_equal(weights[:,rows[0],cols], weights[:,rows[1],cols]):
            raise ValueError('weighted secondary labels require identical preferences')
    else:
        for ri in range(r):
            for row in rows:
                positive = weights[ri,row,cols][weights[ri,row,cols] > 0]
                if positive.size and np.any(positive != positive[0]):
                    raise ValueError('secondary carbon candidate requires uniform positive weights')
    shares = np.divide(labels, target[:,None], out=np.zeros_like(labels), where=target[:,None] > 0)
    common_request = request.copy()
    common_request[:,rows] = 0.
    # Conversions are reserved on the opening stocks. Carbon routing is not used.
    opening, wood, other, selected_common = apply_annual_events(
        state, layout, time, common_request, weights, pwood, phwp, selection_only=True)
    available = opening - selected_common.sum(axis=1)
    area_tol = 1e-10 * max(1., float(opening.sum()))
    if np.any(available < -area_tol):
        raise ValueError('common activity exceeded opening source area')
    available = np.maximum(available[:,cols], 0.) * eligible
    # A slot born at this event is ineligible even if a caller populated it.
    available[:,np.array([slots[i][1] >= time for i in cols])] = 0.
    density = np.divide(wood[0][:,cols] + k[0,:,None]*other[0][:,cols],
                        opening[:,cols], out=np.zeros_like(available), where=opening[:,cols] > 0)
    capacity = (available*density).sum(axis=-1)
    # Passive soft contribution on the identical eligible opening parcels.
    # The leaf-only coefficient never enters selection or carbon routing.
    other_density = np.divide(other[0][:,cols], opening[:,cols],
                              out=np.zeros_like(available), where=opening[:,cols] > 0)
    capacity_soft = (k[0]-k_leaf[0]) * (available*other_density).sum(axis=-1)
    fraction = np.minimum(1., np.divide(target, capacity, out=np.zeros_like(target), where=capacity > 0))
    selected = available * fraction[:,None]
    if secondary_harvest_contract == WEIGHTED_CANDIDATE:
        selected = np.array([select_secondary_area(available[ri], weights[ri,rows[0],cols],
                            density[ri], target[ri]) for ri in range(r)])
    request[:,rows] = shares * selected.sum(axis=-1)[:,None]
    final_weights = weights.copy()
    for row in rows:
        preference = weights[:,row,cols] if secondary_harvest_contract == WEIGHTED_CANDIDATE else eligible
        final_weights[:,row,cols] = preference * np.array([slots[i][1] < time for i in cols])
    out, ledger = apply_annual_events(state, layout, time, request, final_weights, pwood, phwp)
    common_rows = [i for i in range(e) if i not in rows]
    if not np.allclose(ledger.selected_area.values[:,common_rows], selected_common[:,common_rows], rtol=1e-10, atol=area_tol):
        raise RuntimeError('secondary request changed common activity selection')
    if not np.allclose(ledger.selected_area.values[:,rows][:,:,cols], shares[:,:,None]*selected[:,None,:], rtol=1e-10, atol=area_tol):
        raise RuntimeError('secondary labels differ from shared selection')
    removed_w = ledger.removed_wood.isel(event=rows).sum('event').values[0]
    removed_f = ledger.removed_other.isel(event=rows).sum('event').values[0]
    achieved = removed_w + k[0]*removed_f
    if not np.allclose(achieved, fraction*capacity, rtol=1e-10, atol=1e-10):
        raise RuntimeError('actual W/F routing differs from projected delivery')
    values = dict(request=target, capacity=capacity, capacity_soft=capacity_soft,
                  removed_soft=(k[0]-k_leaf[0])*removed_f, achieved=achieved,
                  shortfall=np.maximum(target-achieved,0.), removed_W=removed_w,
                  removed_F=removed_f, area=selected.sum(axis=-1),
                  HWP=ledger.hwp_added.isel(event=rows).sum(('event','box_hwp')).values[0],
                  CWD=ledger.cwd_added.isel(event=rows).sum('event').values[0],
                  soil=ledger.soil_added.isel(event=rows).sum('event').values[0])
    for name,value in values.items():
        ledger['sec_agb_'+name] = (('config','reg_land'),value[None,:])
        ledger['sec_agb_'+name].attrs['units'] = 'Mha' if name == 'area' else 'PgC'
    ledger = ledger.assign_coords(secondary_harvest_label=list(secondary_events))
    ledger['sec_agb_input_area'] = (('config','reg_land','secondary_harvest_label'), np.asarray(event_area)[:,rows][None,:,:])
    ledger.sec_agb_input_area.attrs['units'] = 'Mha'
    ledger['secondary_label_share'] = (('reg_land','secondary_harvest_label'), shares)
    ledger['secondary_label_request'] = (('reg_land','secondary_harvest_label'), labels)
    ledger.secondary_label_share.attrs['units'] = '1'
    ledger.secondary_label_request.attrs['units'] = 'PgC'
    ledger.attrs.update(harvest_contract=secondary_harvest_contract, harvest_projection=projection,
                        projection_status='Operational working projection; not independently identified organs',
                        label_rule='Original gross-carbon request fractions of one common selected matrix; no distinct ages',
                        request_scope='Secondary repeated harvest only; no conversion coproduct subtraction')
    return out, ledger

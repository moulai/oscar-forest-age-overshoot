"""One actual HWP target along a fixed-share, opposite-age-end area path.

This is the minimum-area generalized inverse of an explicit rank path. Zero
wood-yield parcels remain on that path. Actual transfers use the existing
annual event kernel; no alternative carbon routing is introduced here.
"""
import numpy as np

from .regional_forest_age import apply_annual_events, validate_purpose_rank
from .regional_forest_age_allocation import select_ranked_area
from .regional_forest_management import _managed_rows


def select_rank_hwp_area(area, ages, eligibility, young_share, demand, yield_density):
    """Return youngest/oldest selected area and the nonnegative unmet request.

    Area and yield_density must have compatible area/carbon units. At a fixed
    young_share p, the requests are p*H and (1-p)*H. Equal ages share selected
    area proportionally, including parcels with different actual yields.
    The scalar H is the smallest value attaining the reachable carbon target;
    the left end of a zero-yield plateau is selected. No yield cutoff is used.
    """
    area, ages, density = [np.asarray(value, dtype=float) for value in (area, ages, yield_density)]
    eligible = np.asarray(eligibility)
    if (area.ndim != 1 or ages.shape != area.shape or density.shape != area.shape
            or eligible.shape != area.shape or eligible.dtype != np.dtype(bool)):
        raise ValueError('rank HWP selection requires equally shaped source vectors and Boolean eligibility')
    if any(not np.all(np.isfinite(value)) or np.any(value < 0) for value in (area, ages, density)):
        raise ValueError('rank source areas, ages and yields must be finite and nonnegative')
    if (np.ndim(young_share) or not np.isfinite(young_share) or not 0 <= young_share <= 1
            or np.ndim(demand) or not np.isfinite(demand) or demand < 0):
        raise ValueError('young_share must be in [0,1] and demand a finite nonnegative scalar')
    active = eligible & (area > 0)
    selected = np.zeros((2, area.size))
    if demand == 0 or not np.any(active):
        return selected, float(demand)
    _, group = np.unique(ages[active], return_inverse=True)
    grouped_area = np.bincount(group, weights=area[active])
    grouped_carbon = np.bincount(group, weights=area[active] * density[active])
    total_area = float(grouped_area.sum())
    young_edges = np.r_[0., np.cumsum(grouped_area)]
    old_edges = np.r_[0., np.cumsum(grouped_area[::-1])]
    young_carbon = np.r_[0., np.cumsum(grouped_carbon)]
    old_carbon = np.r_[0., np.cumsum(grouped_carbon[::-1])]
    # Every change in marginal yield occurs when either end crosses one of
    # these real-age groups. There is no iterative optimizer or age threshold.
    points = [np.array([0., total_area])]
    if young_share > 0:
        points.append(young_edges / young_share)
    if young_share < 1:
        points.append(old_edges / (1. - young_share))
    points = np.unique(np.concatenate(points))
    points = points[(points >= 0) & (points <= total_area)]
    delivery = (np.interp(young_share * points, young_edges, young_carbon)
                + np.interp((1. - young_share) * points, old_edges, old_carbon))
    capacity = float(delivery[-1])
    target = min(float(demand), capacity)
    if target == 0:
        return selected, float(demand)
    upper = int(np.searchsorted(delivery, target, side='left'))
    if delivery[upper] == target:
        total_selected = float(points[upper])
    else:
        lower = upper - 1
        total_selected = float(points[lower] + (target - delivery[lower]) *
                               (points[upper]-points[lower]) / (delivery[upper]-delivery[lower]))
    requests = total_selected * np.array([young_share, 1. - young_share])
    selected = select_ranked_area(area, ages, np.broadcast_to(eligible, (2, area.size)),
                                  requests, ('youngest', 'oldest'))
    return selected, max(float(demand) - float(np.sum(selected * density)), 0.)


def apply_rank_hwp_target_events(state, layout, time, event_area, weights, pwood,
                                phwp, target_q, *, managed_events, purpose_rank):
    """Apply an actual HWP-carbon request using a fixed-share rank area path.

    event_area contains this branch's declared young/mature share, including
    any prior experimental share change. A zero managed total stays inactive.
    Other activities retain their original areas and are evaluated against
    this branch's opening stocks. New incoming parcels cannot supply harvest.
    Exactly one original member is supported, as in the existing HWP helper.
    """
    if len(layout.members) != 1:
        raise ValueError('rank HWP-target mode requires one original parameter member')
    ranked_source = validate_purpose_rank(layout, purpose_rank)
    rows, source = _managed_rows(layout, managed_events)
    if ranked_source != source or set(managed_events) != set(purpose_rank):
        raise ValueError('rank target activities must match the explicit purpose_rank mapping')
    directions = [purpose_rank[name] for name in managed_events]
    young_row = directions.index('youngest')
    request = np.array(event_area, dtype=float, copy=True)
    weights = np.asarray(weights, dtype=float)
    target = np.asarray(target_q, dtype=float)
    r, e, s = len(layout.regions), len(layout.events), len(layout.slots)
    if request.shape != (r, e) or not np.all(np.isfinite(request)) or np.any(request < 0):
        raise ValueError('event_area must be a finite nonnegative region-event array')
    if weights.shape != (r, e, s) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('weights must be a finite nonnegative region-event-source array')
    if target.shape != (r,) or not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError('target_q must be a finite nonnegative regional PgC vector')
    columns = [i for i, (name, _) in enumerate(layout.slots) if name == source]
    managed_weights = weights[:, rows][:, :, columns]
    for region_weights in managed_weights:
        if not np.array_equal(region_weights[0] > 0, region_weights[1] > 0):
            raise ValueError('rank target requires identical hard eligibility for both labels')
        for row in region_weights:
            positive = row[row > 0]
            if positive.size and np.any(positive != positive[0]):
                raise ValueError('rank target cannot combine nonuniform positive preferences')
    baseline_total = request[:, rows].sum(axis=1)
    shares = np.divide(request[:, rows], baseline_total[:, None], out=np.zeros((r, 2)),
                       where=baseline_total[:, None] > 0)
    common_request = request.copy()
    common_request[:, rows] = 0.
    _, common = apply_annual_events(state, layout, time, common_request, weights, pwood, phwp,
                                    purpose_rank=purpose_rank)
    common_q = common.hwp_added.sum(('event', 'box_hwp')).values[0]
    opening = common.source_area.values
    available = opening - common.selected_area.sum('event').values
    tolerance_area = 1e-10 * max(1., float(opening.sum()))
    if np.any(available < -tolerance_area):
        raise ValueError('common activities exceeded opening source area')
    available = np.maximum(available, 0.)
    biome = layout.biomes.index(next(item.biome for item in layout.classes if item.class_id == source))
    density = np.divide(common.source_wood.values[0], opening, out=np.zeros_like(opening), where=opening > 0)
    yield_density = density[:, columns] * np.asarray(phwp)[0, :, biome, :].sum(axis=-1)[:, None]
    ages = np.maximum(time - np.asarray([layout.slots[index][1] for index in columns]), 0.)
    expected_selection = np.zeros((r, 2, len(columns)))
    for ri in range(r):
        if baseline_total[ri] > 0:
            selected, _ = select_rank_hwp_area(available[ri, columns], ages, managed_weights[ri, 0] > 0,
                float(shares[ri, young_row]), max(float(target[ri]-common_q[ri]), 0.), yield_density[ri])
            for row, direction in enumerate(directions):
                expected_selection[ri, row] = selected[0 if direction == 'youngest' else 1]
    request[:, rows] = expected_selection.sum(axis=-1)
    # The preview state is discarded. Commit only this final application to
    # the original opening state through the existing shared carbon routing.
    out, ledger = apply_annual_events(state, layout, time, request, weights, pwood, phwp,
                                     purpose_rank=purpose_rank)
    common_rows = [index for index in range(e) if index not in rows]
    if not np.allclose(ledger.selected_area.values[:, common_rows], common.selected_area.values[:, common_rows],
                       rtol=1e-10, atol=tolerance_area):
        raise RuntimeError('rank target changed common-activity allocation')
    if not np.allclose(ledger.selected_area.values[:, rows][:, :, columns], expected_selection,
                       rtol=1e-10, atol=tolerance_area):
        raise RuntimeError('final rank allocation differs from target selection')
    achieved = ledger.hwp_added.sum(('event', 'box_hwp')).values[0]
    managed_q = ledger.hwp_added.isel(event=rows).sum(('event', 'box_hwp')).values[0]
    predicted = common_q + (expected_selection.sum(axis=1) * yield_density).sum(axis=1)
    if not np.allclose(achieved, predicted, rtol=1e-10, atol=1e-10):
        raise RuntimeError('actual routed HWP differs from rank target prediction')
    for name, values in (('hwp_target', target), ('hwp_achieved', achieved),
                         ('hwp_common', achieved-managed_q), ('hwp_managed', managed_q),
                         ('hwp_shortfall', np.maximum(target-achieved, 0.)),
                         ('hwp_excess', np.maximum(achieved-target, 0.))):
        ledger[name] = (('config', 'reg_land'), values[None, :])
        ledger[name].attrs['units'] = 'PgC'
    ledger['baseline_requested_area'] = (('reg_land', 'event'), np.asarray(event_area, dtype=float))
    ledger.baseline_requested_area.attrs['units'] = 'Mha'
    ledger = ledger.assign_coords(managed_event=list(managed_events))
    ledger['managed_label_share'] = (('reg_land', 'managed_event'),
                                     np.where(baseline_total[:, None] > 0, shares, np.nan))
    ledger.managed_label_share.attrs['units'] = '1'
    ledger['managed_activity_available'] = ('reg_land', baseline_total > 0)
    actual_managed = ledger.selected_area.values[:, rows][:, :, columns]
    for suffix, reduction in (('min', np.min), ('max', np.max)):
        boundary = np.full((r, 2), np.nan)
        for ri in range(r):
            for row in range(2):
                used = actual_managed[ri, row] > 0
                if used.any():
                    boundary[ri, row] = reduction(ages[used])
        ledger['managed_source_'+suffix+'_age'] = (('reg_land', 'managed_event'), boundary)
        ledger['managed_source_'+suffix+'_age'].attrs['units'] = 'yr'
    ledger['hwp_target_met'] = (('config', 'reg_land'),
        (np.abs(achieved-target) <= 1e-10 * np.maximum(1., np.maximum(target, achieved)))[None, :])
    ledger.attrs.update(hwp_target_definition='All actual HWP wood-carbon input including common conversion coproducts',
                        managed_label_rule='Declared request-area shares; zero total stays inactive with undefined shares',
                        rank_inverse='Minimum total area on fixed-share rank path; zero-yield sources retained',
                        target_tolerance_PgC=1e-10, target_relative_tolerance=1e-10)
    return out, ledger

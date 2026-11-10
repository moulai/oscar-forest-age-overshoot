"""Explicit age preferences and one actual HWP-carbon delivery constraint.

These policies consume prepared source/event identities, fixed regional
maturity functions, and actual opening stocks. They do not read raw land-use
products, infer age thresholds from activity names, or run an experiment.
"""
import numpy as np

from .regional_forest_age import apply_annual_events
from .regional_forest_age_allocation import select_area


def _managed_rows(layout, event_ids):
    """Check that two named labels describe the same cohort harvest process."""
    if len(event_ids) != 2 or len(set(event_ids)) != 2:
        raise ValueError("exactly two distinct managed event identities required")
    identities = [event.event_id for event in layout.events]
    if any(name not in identities for name in event_ids):
        raise ValueError("managed event identity missing from layout")
    rows = [identities.index(name) for name in event_ids]
    events = [layout.events[row] for row in rows]
    source = events[0].source_class
    if source not in layout.forest_classes or any(
            event.source_class != source or event.target_class != source
            or event.kind != "harvest" or event.vegetation_action != "clear"
            or event.age_action != "reset" for event in events):
        raise ValueError("managed labels must clear and reset one shared cohort class")
    return rows, source


def management_weights(layout, time, maturity=None, *, mode="area_proportional",
                       managed_events=(), strength=1., eligibility=None):
    """Return (region, event, source_slot) weights, separate from eligibility.

    ``maturity(ages)`` returns fixed S in [0, 1], with shape (region, birth).
    Ages are actual years. S must be prepared independently of parameter
    members, current carbon stocks, and the age-production switch. It is not
    evaluated for primary forest, which has no assigned age.

    Modes: area_proportional gives one; purpose_maturity gives 1+k*S to the
    first named event and 1+k*(1-S) to the second; secondary_maturity gives
    1+k*S to both. k=1 is an explicit bounded 2:1 experiment, not an estimate.
    Boolean eligibility is a hard constraint and cannot be relaxed by k.
    """
    modes = ("area_proportional", "purpose_maturity", "secondary_maturity")
    if mode not in modes or not np.isfinite(time):
        raise ValueError("unknown policy mode or invalid time")
    if np.ndim(strength) or not np.isfinite(strength) or strength < 0:
        raise ValueError("preference strength must be finite and nonnegative")
    shape = (len(layout.regions), len(layout.events), len(layout.slots))
    weights = np.ones(shape)
    if mode != "area_proportional":
        rows, source = _managed_rows(layout, managed_events)
        if maturity is None:
            raise ValueError("maturity policy requires a fixed regional S evaluator")
        # Future reserved slots are empty; evaluating their shape at zero does
        # not create area, birth times, or ages for any unaged source.
        ages = np.maximum(time - np.asarray(layout.births), 0.)
        s = np.asarray(maturity(ages), dtype=float)
        if s.shape != (len(layout.regions), len(layout.births)) or not np.all(
                np.isfinite(s)) or np.any((s < 0) | (s > 1)):
            raise ValueError("S must have region-by-birth shape and lie in [0, 1]")
        columns = [i for i, (name, _) in enumerate(layout.slots) if name == source]
        weights[:, rows[0], columns] = 1. + strength * s
        weights[:, rows[1], columns] = 1. + strength * (
            1. - s if mode == "purpose_maturity" else s)
    if eligibility is not None:
        eligible = np.asarray(eligibility)
        if eligible.shape != shape or eligible.dtype != np.dtype(bool):
            raise ValueError("eligibility must be a boolean region-event-source array")
        weights *= eligible
    return weights


def apply_hwp_target_events(state, layout, time, event_area, weights, pwood,
                            phwp, target_q, *, managed_events):
    """Apply one annual event with a single actual HWP wood-carbon target.

    The target is (region,) PgC per annual event, summed across every event
    and HWP product. Conversion coproducts and unchanged activities contribute
    first. Only the two named harvest labels may change their areas, using
    the remaining *opening* source parcels and their actual W/A * sum(p_hwp).

    The function requires one parameter member and identical weights and
    eligibility for the managed labels. Their original area shares preserve
    reporting labels, without interpreting them as distinct wood products.
    If their total baseline area is zero, neither label is activated: any
    residual target is reported as a shortfall. Other harvest activities may
    not compete for this managed source in this minimal implementation.

    Returns (next_state, ledger). A discarded pure common-activity preview
    never becomes the input state. The final call starts from the original
    state, and its actual ledger is the sole source of achieved delivery.
    """
    if len(layout.members) != 1:
        raise ValueError("HWP-target mode requires one original parameter member")
    rows, source = _managed_rows(layout, managed_events)
    if any(event.kind == "harvest" and event.source_class == source
           and i not in rows for i, event in enumerate(layout.events)):
        raise ValueError("other harvest activities compete for the managed source")
    target = np.asarray(target_q, dtype=float)
    if target.shape != (len(layout.regions),) or not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("target_q must be a finite nonnegative regional PgC vector")
    request = np.array(event_area, dtype=float, copy=True)
    weights = np.asarray(weights, dtype=float)
    expected = (len(layout.regions), len(layout.events))
    if request.shape != expected or not np.all(np.isfinite(request)) or np.any(request < 0):
        raise ValueError("event_area must be a finite nonnegative region-event array")
    if weights.shape != expected + (len(layout.slots),):
        raise ValueError("weights must have region-event-source shape")
    cols = [i for i, (name, _) in enumerate(layout.slots) if name == source]
    if not np.array_equal(weights[:, rows[0], cols], weights[:, rows[1], cols]):
        raise ValueError("managed labels must have identical preferences and eligibility")
    baseline = request[:, rows].sum(axis=1)
    shares = np.divide(request[:, rows], baseline[:, None],
                       out=np.zeros((len(layout.regions), 2)), where=baseline[:, None] > 0)
    common_request = request.copy()
    common_request[:, rows] = 0.
    _, common = apply_annual_events(state, layout, time, common_request, weights, pwood, phwp)
    common_q = common.hwp_added.sum(("event", "box_hwp")).values[0]
    # Read the *opening* sources from the preview ledger, not its post-event
    # state: new or retained incoming forest can never supply this event.
    opening = common.source_area.values
    available = opening - common.selected_area.sum("event").values
    tolerance_area = 1e-10 * max(1., float(opening.sum()))
    if np.any(available < -tolerance_area):
        raise ValueError("common activities exceeded opening source area")
    available = np.maximum(available, 0.)
    biome = next(c.biome for c in layout.classes if c.class_id == source)
    bi = layout.biomes.index(biome)
    wood_density = np.divide(common.source_wood.values[0], opening,
                             out=np.zeros_like(opening), where=opening > 0)
    yield_density = wood_density[:, cols] * np.asarray(phwp)[0, :, bi, :].sum(axis=-1)[:, None]
    selected = np.zeros((len(layout.regions), len(cols)))
    for ri in range(len(layout.regions)):
        if baseline[ri] > 0:
            selected[ri], _ = select_area(
                available[ri, cols], weights[ri, rows[0], cols],
                max(0., target[ri] - common_q[ri]), yield_density=yield_density[ri])
    request[:, rows] = shares * selected.sum(axis=1)[:, None]
    # Pure reapplication from the SAME opening state commits the annual
    # route once; no preview state or preview product stock is reused.
    final_weights = weights.copy()
    for row in rows:
        final_weights[:, row, cols] *= yield_density > 0
    out, ledger = apply_annual_events(state, layout, time, request, final_weights, pwood, phwp)
    common_rows = [i for i in range(len(layout.events)) if i not in rows]
    if not np.allclose(ledger.selected_area.values[:, common_rows],
                       common.selected_area.values[:, common_rows], rtol=1e-10, atol=tolerance_area):
        raise RuntimeError("HWP target changed common-activity allocation")
    expected_selection = shares[:, :, None] * selected[:, None, :]
    if not np.allclose(ledger.selected_area.values[:, rows][:, :, cols],
                       expected_selection, rtol=1e-10, atol=tolerance_area):
        raise RuntimeError("final area allocation differs from HWP-target selection")
    achieved = ledger.hwp_added.sum(("event", "box_hwp")).values[0]
    managed_q = ledger.hwp_added.isel(event=rows).sum(("event", "box_hwp")).values[0]
    predicted = common_q + (selected * yield_density).sum(axis=1)
    if not np.allclose(achieved, predicted, rtol=1e-10, atol=1e-10):
        raise RuntimeError("actual routed HWP carbon differs from target allocation")
    for name, values in (("hwp_target", target), ("hwp_achieved", achieved),
                         ("hwp_common", achieved - managed_q), ("hwp_managed", managed_q),
                         ("hwp_shortfall", np.maximum(target - achieved, 0.)),
                         ("hwp_excess", np.maximum(achieved - target, 0.))):
        ledger[name] = (("config", "reg_land"), values[None, :])
        ledger[name].attrs["units"] = "PgC"
    ledger["baseline_requested_area"] = (("reg_land", "event"), np.asarray(event_area, dtype=float))
    ledger.baseline_requested_area.attrs["units"] = "Mha"
    ledger = ledger.assign_coords(managed_event=list(managed_events))
    ledger["managed_label_share"] = (("reg_land", "managed_event"), shares)
    ledger.managed_label_share.attrs["units"] = "1"
    ledger["managed_activity_available"] = ("reg_land", baseline > 0)
    ledger["hwp_target_met"] = (("config", "reg_land"),
        (np.abs(achieved - target) <= 1e-10 * np.maximum(1., np.maximum(target, achieved)))[None, :])
    ledger.attrs.update(hwp_target_definition="All actual wood carbon routed to all HWP products, including conversion coproducts",
                        managed_label_rule="Baseline requested-area shares; zero total leaves both labels inactive",
                        target_tolerance_PgC=1e-10, target_relative_tolerance=1e-10)
    return out, ledger

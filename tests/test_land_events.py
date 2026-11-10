"""Carbon routing from explicit selected flows, independent of age selection."""
from dataclasses import replace
from math import fsum
import json
import numpy as np
import pytest

from oscar._core._model._modules.forest_age import TwoPoolForestState
from oscar._core._model._modules.land_geometry import (
    LandClass, EventSpec, SourceKey, AreaTable, SelectedArea, RegionalPlan,
)
from oscar._core._model._modules.land_events import LandEventState, EventRouting, apply_events
from oscar._core._model._modules.land_geometry import secondary_stand_harvest

CLASSES = (LandClass("primf", "Forest", "forest_unaged"),
           LandClass("secdf", "Forest", "forest_cohort"),
           LandClass("pltns", "Forest", "forest_cohort"),
           LandClass("range", "Non-Forest", "nonforest"),
           LandClass("c3ann", "Cropland", "nonforest"))
BIOMES = ("Forest", "Non-Forest", "Cropland")
IDS = (101, 309)


def state(*, area=(1., 1.), births=(90., 90.), origins=("range", "cleared"),
          uses=("secdf", "secdf"), carbon=None, nf=(1., 1.), nfcarbon=None,
          primary_area=0., primary=None, cwd=None, soil=None, hwp=None):
    m, k = len(IDS), len(area)
    if carbon is None:
        carbon = np.broadcast_to(np.asarray(area)[None, None, :], (m, 2, k)).copy()
    forest = TwoPoolForestState(100.5, IDS, primary_area,
        np.zeros((m, 2)) if primary is None else primary, births, area, carbon, "R", origins)
    return LandEventState(forest, CLASSES, BIOMES, ("short", "long"), uses,
        nf, np.tile(nf, (m, 1)) if nfcarbon is None else nfcarbon,
        np.zeros((m, 3)) if cwd is None else cwd,
        np.zeros((m, 3)) if soil is None else soil,
        np.zeros((m, 3, 2)) if hwp is None else hwp, 100)


def routing():
    return EventRouting(IDS, BIOMES, ("short", "long"),
                        np.tile([.8, .1, .3], (2, 1)), np.full((2, 3, 2), .25))


def plan_for(s, events, selections):
    """An explicit flow fixture; never decide which age or origin is selected."""
    selections = np.asarray(selections, dtype=float).reshape(len(events), len(s.source_keys))
    keys = s.source_keys
    outgoing = np.array([fsum(col) for col in selections.T])
    closing = {key: max(0., area-out) for key, area, out in zip(keys, s.source_area, outgoing)}
    classes = {c.class_id: c for c in s.classes}
    incoming = {}
    for e, event in enumerate(events):
        for j in np.flatnonzero(selections[e]):
            source = keys[j]
            if classes[event.target_class].role == "nonforest":
                target = SourceKey(event.target_class)
            elif event.age_action == "keep" or secondary_stand_harvest(event):
                target = SourceKey(event.target_class, source.birth_time, source.entry_origin)
            else:
                target = SourceKey(event.target_class, s.next_event_year+.5,
                    "cleared" if event.vegetation_action == "clear" else event.source_class)
            incoming[target] = incoming.get(target, 0.)+selections[e, j]
    for key, amount in incoming.items():
        closing[key] = closing.get(key, 0.)+amount
    e, k = np.nonzero(selections)
    totals = np.array([fsum(row) for row in selections])[None]
    survivors = AreaTable(keys, np.maximum(s.source_area-outgoing, 0.)[None])
    return RegionalPlan((s.forest.region_id,), tuple(events), "proportional", s.next_event_year,
        s.next_event_year+1, s.next_event_year+.5, s.next_event_year+1, s.next_event_year+1,
        AreaTable(keys, s.source_area[None]),
        SelectedArea(np.zeros(len(e), dtype=int), e, k, selections[e, k]),
        AreaTable(tuple(closing), np.array(list(closing.values()))[None]), totals, totals, survivors)


def event(name="h", source="secdf", target="secdf", kind="harvest", action="clear", age="reset"):
    return EventSpec(name, source, target, kind, action, age, "proportional",
                     "source_p_wood" if action == "retain" and source in ("range", "c3ann") and target in ("secdf", "pltns") else "none")


def apply(s, events, selections):
    p = plan_for(s, events, selections)
    out, flux = apply_events(s, p, routing())
    np.testing.assert_allclose(out.total_carbon, s.total_carbon, rtol=2e-14, atol=0)
    return out, flux, p


def test_selected_origin_actual_wf_and_position_independence():
    carbon = np.array([[[10., 0.], [2., 0.]], [[20., 0.], [3., 0.]]])
    s = state(carbon=carbon)
    out, flux, p = apply(s, [event()], [[0, 1, 0, 0, 0]])
    np.testing.assert_array_equal(flux.removed[:, :, 0], carbon[:, :, 0])
    np.testing.assert_array_equal(flux.hwp_added[:, 0].sum(axis=1), [5, 10])
    np.testing.assert_array_equal(out.forest.total_carbon, 0)
    perm = [1, 0]
    shuffled = replace(s, forest=replace(s.forest, birth_time=s.forest.birth_time[perm],
        secondary_area=s.forest.secondary_area[perm], secondary_carbon=carbon[:, :, perm],
        entry_origin=tuple(s.forest.entry_origin[i] for i in perm)))
    same, ledger = apply_events(shuffled, p, routing())
    assert same.snapshot() == out.snapshot()
    np.testing.assert_array_equal(ledger.removed, flux.removed)


def test_keep_coalesces_existing_target_and_preserves_origin():
    s = state(origins=("range", "range"), uses=("secdf", "pltns"))
    keep = event("keep", "secdf", "pltns", "conversion", "retain", "keep")
    out, flux, _ = apply(s, [keep], [[0, .5, 0, 0, 0]])
    assert out.cohort_class == ("secdf", "pltns")
    assert out.forest.entry_origin == ("range", "range")
    np.testing.assert_array_equal(out.forest.secondary_area, [.5, 1.5])
    np.testing.assert_array_equal(out.forest.secondary_carbon[:, :, 1], 1.5)
    np.testing.assert_array_equal(flux.hwp_added, 0)
    with pytest.raises(ValueError, match="duplicate complete"):
        replace(s, cohort_class=("secdf", "secdf"))


def test_tiny_keep_survives_complete_target_clearance():
    tiny = 1e-250
    s = state(area=(tiny, 1.), origins=("range", "range"), uses=("secdf", "pltns"))
    events = [event("keep", "secdf", "pltns", "conversion", "retain", "keep"),
              event("clear", "pltns", "pltns")]
    out, _, _ = apply(s, events, [[0, tiny, 0, 0, 0], [0, 0, 1, 0, 0]])
    ix = out.source_keys.index(SourceKey("pltns", 90., "range"))-1
    assert out.forest.secondary_area[ix] == tiny
    np.testing.assert_array_equal(out.forest.secondary_carbon[:, :, ix], tiny)


def test_frozen_nonforest_donor_retention_and_incoming_not_reused():
    s = state(nfcarbon=np.tile([10., 2.], (2, 1)))
    events = [event("establish", "range", "secdf", "conversion", "retain", "reset"),
              event("exit", "secdf", "range", "conversion", "retain", "none")]
    out, flux, _ = apply(s, events, [[0, 0, 0, .5, 0], [0, .5, 0, 0, 0]])
    np.testing.assert_array_equal(flux.retained[:, :, 0], [[.5, 4.5], [.5, 4.5]])
    assert SourceKey("secdf", 100.5, "range") in out.source_keys
    np.testing.assert_array_equal(out.nonforest_carbon[:, 0], 6.)
    np.testing.assert_array_equal(flux.hwp_added, 0)


def test_old_dead_pools_move_new_residue_does_not_feed_batch_hwp_stays_source():
    s = state(area=(1.,), births=(90.,), origins=("cleared",), uses=("secdf",),
              cwd=np.tile([10., 20., 0.], (2, 1)), soil=np.tile([30., 40., 0.], (2, 1)))
    events = [event("exit", "secdf", "c3ann", "conversion", "clear", "none"),
              event("enter", "range", "secdf", "conversion", "clear", "reset")]
    out, flux, _ = apply(s, events, [[0, 1, 0, 0], [0, 0, 1, 0]])
    np.testing.assert_array_equal(flux.dead_transferred[:, :, 0], [[10, 30], [10, 30]])
    np.testing.assert_array_equal(flux.dead_transferred[:, :, 1], [[20, 40], [20, 40]])
    np.testing.assert_allclose(out.cwd, np.tile([20.05, 0, 10.5], (2, 1)))
    np.testing.assert_allclose(out.soil, np.tile([40.9, 0, 31.], (2, 1)))
    np.testing.assert_array_equal(out.hwp[:, 0].sum(axis=1), .5)
    np.testing.assert_array_equal(out.hwp[:, 2], 0)


def test_zero_year_consumed_once_and_v2_roundtrip_rejects_legacy():
    s = state()
    out, flux, p = apply(s, [], np.empty((0, len(s.source_keys))))
    assert out.next_event_year == 101 and not flux.event_ids
    with pytest.raises(ValueError, match="cursor"):
        apply_events(out, p, routing())
    loaded = LandEventState.from_snapshot(json.loads(json.dumps(out.snapshot())))
    assert loaded.snapshot() == out.snapshot()
    bad = out.snapshot(); bad["schema"] = "land-events-v1"
    with pytest.raises(ValueError, match="schema"):
        LandEventState.from_snapshot(bad)
    bad = out.snapshot(); bad["forest"]["schema"] = "forest-two-pool-v1"
    with pytest.raises(ValueError, match="schema"):
        LandEventState.from_snapshot(bad)


def test_wrong_origin_opening_and_zero_area_stock_rejected():
    s = state()
    p = plan_for(s, [event()], [[0, .5, 0, 0, 0]])
    wrong = replace(s, forest=replace(s.forest, entry_origin=("c3ann", "cleared")))
    with pytest.raises(ValueError, match="opening"):
        apply_events(wrong, p, routing())
    with pytest.raises(ValueError, match="zero secondary"):
        replace(s.forest, secondary_area=[0, 1])
    with pytest.raises(ValueError, match="zero nonforest"):
        replace(s, nonforest_area=[0, 1])
    with pytest.raises(ValueError, match="entry_origin"):
        replace(s.forest, entry_origin=("unknown", "cleared"))


def test_regional_plan_rejects_malformed_sparse_budget_and_closing():
    s = state(); p = plan_for(s, [event()], [[0, .5, 0, 0, 0]])
    with pytest.raises(ValueError, match="duplicate"):
        SelectedArea([0, 0], [0, 0], [1, 1], [.2, .3])
    with pytest.raises(ValueError, match="budget"):
        replace(p, selected=SelectedArea([0], [0], [1], [2.]))
    with pytest.raises(ValueError, match="calendar"):
        replace(p, event_time=100.25)
    malformed = replace(p, closing=AreaTable(p.closing.keys, p.closing.area*2))
    with pytest.raises(ValueError, match="closing"):
        apply_events(s, malformed, routing())
    assert s.next_event_year == 100


def test_unresolved_positive_event_and_primary_creation_rejected():
    s = state()
    unknown = event("unknown", "secdf", "pltns", "conversion", "unresolved", "reset")
    p = replace(plan_for(s, [replace(unknown, vegetation_action="clear")], [[0, .5, 0, 0, 0]]), events=(unknown,))
    with pytest.raises(ValueError, match="unresolved"):
        apply_events(s, p, routing())
    untouched, _, _ = apply(s, [unknown], [[0, 0, 0, 0, 0]])
    assert untouched.next_event_year == 101


def test_primary_removal_and_member_routing_alignment():
    s = state(primary_area=1., primary=np.array([[1., 9.], [2., 8.]]))
    out, flux, p = apply(s, [event(source="primf")], [[.5, 0, 0, 0, 0]])
    np.testing.assert_array_equal(flux.removed[:, :, 0], [[.5, 4.5], [1., 4.]])
    r = routing()
    reverse = replace(r, member_ids=IDS[::-1], p_wood=r.p_wood[::-1], p_hwp=r.p_hwp[::-1])
    same, _ = apply_events(s, p, reverse)
    assert same.snapshot() == out.snapshot()
    with pytest.raises(ValueError, match="differ"):
        apply_events(s, p, replace(r, member_ids=(101, 999)))


def test_shared_plan_uses_each_regions_actual_density():
    original = state(carbon=np.array([[[10., 0.], [2., 0.]], [[20., 0.], [3., 0.]]]))
    base = plan_for(original, [event()], [[0, 1., 0, 0, 0]])
    weights = np.array([.25, .75])
    shared = replace(base, region_ids=("R", "Q"),
        opening=AreaTable(base.opening.keys, weights[:, None]*base.opening.area),
        surviving=AreaTable(base.surviving.keys, weights[:, None]*base.surviving.area),
        closing=AreaTable(base.closing.keys, weights[:, None]*base.closing.area),
        selected=SelectedArea([0, 1], [0, 0], [1, 1], weights),
        requested_area=weights[:, None], executed_area=weights[:, None],
        initial_land_area=weights*base.initial_land_area[0])
    for name, weight, density in zip(("R", "Q"), weights, (1., 2.)):
        forest = replace(original.forest, region_id=name,
            secondary_area=original.forest.secondary_area*weight,
            secondary_carbon=original.forest.secondary_carbon*weight*density)
        s = replace(original, forest=forest, nonforest_area=original.nonforest_area*weight,
                    nonforest_carbon=original.nonforest_carbon*weight*density)
        out, flux = apply_events(s, shared, routing())
        np.testing.assert_array_equal(flux.removed[:, :, 0], original.forest.secondary_carbon[:, :, 0]*weight*density)
        np.testing.assert_allclose(out.total_carbon, s.total_carbon, rtol=2e-14, atol=0)


def test_decimal_exhaustion_keeps_each_selected_event_and_closes_carbon():
    s = state(area=(.1, .2, .7), births=(80., 85., 90.),
              origins=("cleared",)*3, uses=("secdf",)*3)
    events = [event("a"), event("b"), event("c")]
    selected = np.zeros((3, len(s.source_keys)))
    selected[0, 1:4] = [.04, .08, .28]
    selected[1, 1:4] = [.04, .08, .28]
    selected[2, 1:4] = [.02, .04, .14]
    out, flux, _ = apply(s, events, selected)
    np.testing.assert_allclose(flux.executed_area, [.4, .4, .2], rtol=3e-15, atol=0)
    np.testing.assert_allclose(out.forest.secondary_area.sum(), 1., rtol=3e-15, atol=0)
    assert np.max(abs(flux.budget_residual)) < 1e-14

"""Hand-computable geometry oracles; no carbon integration or real history."""
from dataclasses import replace
import json

import numpy as np
import pytest

from oscar._core._model._modules.land_geometry import (
    AreaShortage, ENTRY_ORIGINS, EventSpec, GeometryState, LandClass, SourceKey,
    advance_geometry,
)


CLASSES = (
    LandClass("primf", "Forest", "forest_unaged"),
    LandClass("primn", "Nonforest", "nonforest"),
    LandClass("secdf", "Forest", "forest_cohort"),
    LandClass("secdn", "Nonforest", "nonforest"),
    LandClass("urban", "Urban", "nonforest"),
    *(LandClass(c, "Crop", "nonforest") for c in ("c3ann", "c4ann", "c3per", "c4per", "c3nfx")),
    LandClass("pastr", "Pasture", "nonforest"),
    LandClass("range", "Nonforest", "nonforest"),
    LandClass("pltns", "Forest", "forest_cohort"),
)


def state(forest=(), *, fixed=None, cells=1, year=100, strategy="proportional"):
    columns = {SourceKey(c.class_id): np.zeros(cells) for c in CLASSES if c.role != "forest_cohort"}
    for name, area in (fixed or {}).items():
        columns[SourceKey(name)] = np.broadcast_to(area, (cells,))
    for name, birth, origin, area in forest:
        columns[SourceKey(name, birth, origin)] = np.broadcast_to(area, (cells,))
    return GeometryState(CLASSES, tuple(range(cells)), tuple(columns),
                         np.column_stack(list(columns.values())), year, strategy)


def harvest(name="secmf_harv", selection="oldest", source="secdf", target="secdf"):
    age = "reset" if target in ("secdf", "pltns") else "none"
    return EventSpec(name, source, target, "harvest", "clear", age, selection)


def conversion(name, source, target, vegetation="clear", age=None):
    if age is None:
        age = "reset" if target in ("secdf", "pltns") else "none"
    retained_split = "source_p_wood" if (vegetation == "retain" and source in ENTRY_ORIGINS[1:]
                                                   and target in ("secdf", "pltns")) else "none"
    return EventSpec(name, source, target, "conversion", vegetation, age, "proportional", retained_split)


def run(s, specs, demand, *, weights=None):
    if weights is None:
        weights = np.ones((len(s.cell_ids), 1))
    return advance_geometry(s, specs, demand, year=s.next_year,
                            region_ids=tuple(f"R{i}" for i in range(weights.shape[1])),
                            region_fraction=weights)


def area(s, name, birth=None, origin=None):
    key = SourceKey(name, birth, origin)
    return s.area[:, s.keys.index(key)] if key in s.keys else np.zeros(len(s.cell_ids))


def selected(plan, event, key, region=0):
    event_index = [e.event_id for e in plan.events].index(event)
    source_index = plan.opening.keys.index(key)
    rows = ((plan.selected.region_index == region) & (plan.selected.event_index == event_index)
            & (plan.selected.source_index == source_index))
    return plan.selected.area[rows].sum()


def test_two_cell_local_selection_differs_from_aggregate_first_03bk():
    # Two 1-ha stands, aged 1 and 99 at the event, with 2 and 100 MgC/ha.
    young, old = SourceKey("secdf", 99.5, "cleared"), SourceKey("secdf", 1.5, "cleared")
    s = state([("secdf", 99.5, "cleared", [1e-6, 0]),
               ("secdf", 1.5, "cleared", [0, 1e-6])], cells=2)
    out, local = run(s, [harvest()], [[1e-6], [0]])
    aggregated = GeometryState(s.classes, (0,), s.keys, s.area.sum(axis=0)[None], 100, s.strategy)
    aggregate_out, first = run(aggregated, [harvest()], [[1e-6]])
    np.testing.assert_allclose([selected(local, "secmf_harv", young), selected(local, "secmf_harv", old)],
                               [1e-6, 0], rtol=1e-15, atol=0)
    np.testing.assert_allclose([selected(first, "secmf_harv", young), selected(first, "secmf_harv", old)],
                               [.5e-6, .5e-6], rtol=1e-15, atol=0)
    removals = [sum(selected(p, "secmf_harv", k) * 1e6 * d for k, d in [(young, 2), (old, 100)])
                for p in (local, first)]
    np.testing.assert_allclose(removals, [2, 51])
    np.testing.assert_allclose(102 - np.array(removals), [100, 51])
    def mean_age(st):
        return sum(st.area[:, i].sum() * (100.5 - k.birth_time)
                   for i, k in enumerate(st.keys) if k.birth_time is not None) / 2e-6
    # Selection still differs. Age does not, because the cut stand is not reborn.
    np.testing.assert_allclose([mean_age(out), mean_age(aggregate_out)], [50, 50])


def test_same_birth_different_origin_is_not_birth_only_carbon_mixing():
    s = state([("secdf", 90.5, "c3ann", [1, 0]),
               ("secdf", 90.5, "cleared", [0, 1])], cells=2)
    _, plan = run(s, [harvest()], [[1], [0]])
    retained = selected(plan, "secmf_harv", SourceKey("secdf", 90.5, "c3ann"))
    zero = selected(plan, "secmf_harv", SourceKey("secdf", 90.5, "cleared"))
    assert (retained, zero) == (1, 0)
    assert 10 * retained + 0 * zero == 10  # Externally specified density oracle.
    assert (10 + 0) / 2 == 5  # Incorrect birth-only answer.


@pytest.mark.parametrize("strategy,old_take,young_take,survivors", [
    ("land_use_specific", [.3, 0], [0, .2], [.4, .4]),
    ("proportional", [.15, .15], [.1, .1], [.4, .4]),
])
def test_proportional_conversion_then_joint_two_tail_harvest(strategy, old_take, young_take, survivors):
    s = state([("secdf", 1.5, "cleared", .5), ("secdf", 90.5, "cleared", .5)], strategy=strategy)
    specs = [harvest("young", "youngest"), conversion("crop", "secdf", "c3ann"), harvest("old")]
    out, plan = run(s, specs, [[.2, .2, .3]])
    keys = [SourceKey("secdf", b, "cleared") for b in (1.5, 90.5)]
    np.testing.assert_allclose([selected(plan, "crop", k) for k in keys], [.1, .1])
    np.testing.assert_allclose([selected(plan, "old", k) for k in keys], old_take, atol=1e-16)
    np.testing.assert_allclose([selected(plan, "young", k) for k in keys], young_take, atol=1e-16)
    np.testing.assert_allclose([area(out, "secdf", b, "cleared")[0] for b in (1.5, 90.5)], survivors)
    np.testing.assert_allclose(area(out, "secdf", 100.5, "cleared"), [0.])
    np.testing.assert_allclose(area(out, "c3ann"), [.2])
    np.testing.assert_allclose(out.area.sum(axis=1), [1])


def test_same_age_origins_and_tied_events_share_proportionally():
    s = state([("secdf", 1.5, "cleared", .3), ("secdf", 1.5, "c3ann", .1),
               ("secdf", 90.5, "cleared", .6)], strategy="land_use_specific")
    out, plan = run(s, [harvest("b"), harvest("a")], [[.1, .1]])
    for event in ("a", "b"):
        np.testing.assert_allclose([selected(plan, event, SourceKey("secdf", 1.5, origin))
                                   for origin in ("cleared", "c3ann")], [.075, .025])
    np.testing.assert_allclose(area(out, "secdf", 1.5, "cleared"), [.3])
    np.testing.assert_allclose(area(out, "secdf", 1.5, "c3ann"), [.1])


@pytest.mark.parametrize("specs,demands", [
    ([conversion("a", "secdf", "pltns"), conversion("b", "pltns", "secdf")], [[1, 1]]),
    ([harvest("a"), harvest("b")], [[.6, .6]]),
    ([conversion("a", "secdf", "c3ann"), harvest("b")], [[.4, .7]]),
])
def test_full_opening_budget_rejects_inflows_rebirth_and_partial_commit(specs, demands):
    s = state([("secdf", 1.5, "cleared", 1)])
    before = json.dumps(s.snapshot(), sort_keys=True)
    with pytest.raises(AreaShortage):
        run(s, specs, demands)
    assert json.dumps(s.snapshot(), sort_keys=True) == before
    assert not s.area.flags.writeable


def test_exact_conversion_plus_harvest_source_exhaustion():
    s = state([("secdf", 1.5, "cleared", 1)])
    out, plan = run(s, [conversion("a", "secdf", "c3ann"), harvest("b")], [[.4, .6]])
    np.testing.assert_allclose(area(out, "secdf", 1.5, "cleared"), [.6])
    np.testing.assert_allclose(area(out, "c3ann"), [.4])
    np.testing.assert_allclose(area(out, "secdf", 100.5, "cleared"), [0.])
    np.testing.assert_allclose(plan.executed_area, [[.4, .6]])


def test_keep_merges_with_surviving_existing_complete_key_and_clear_rebirth():
    s = state([("secdf", 90.5, "c3ann", .4), ("pltns", 90.5, "c3ann", .2),
               ("pltns", 90.5, "cleared", .3)])
    specs = [conversion("keep", "secdf", "pltns", "retain", "keep"),
             harvest("clear", "oldest", "pltns", "pltns")]
    out, _ = run(s, specs, [[.2, .1]])
    assert len(out.keys) == len(set(out.keys))
    # Existing same-age plantations share clearance .04/.06, then keep adds .2.
    np.testing.assert_allclose(area(out, "pltns", 90.5, "c3ann"), [.36])
    np.testing.assert_allclose(area(out, "pltns", 90.5, "cleared"), [.24])
    np.testing.assert_allclose(area(out, "pltns", 100.5, "cleared"), [.1])
    np.testing.assert_allclose(area(out, "secdf", 90.5, "c3ann"), [.2])
    assert out.next_year - 100.5 == .5


def test_nonforest_clear_retain_and_nonforest_harvest_keep_distinct_semantics():
    s = state(fixed={"c3ann": 1, "primn": .4, "secdn": .3})
    specs = [conversion("clear", "c3ann", "secdf"),
             conversion("retain", "c3ann", "secdf", "retain"),
             harvest("primn_harv", "proportional", "primn", "secdn"),
             harvest("secnf_harv", "proportional", "secdn", "secdn")]
    out, plan = run(s, specs, [[.2, .3, .1, .2]])
    np.testing.assert_allclose(area(out, "secdf", 100.5, "cleared"), [.2])
    np.testing.assert_allclose(area(out, "secdf", 100.5, "c3ann"), [.3])
    np.testing.assert_allclose(area(out, "c3ann"), [.5])
    np.testing.assert_allclose(area(out, "primn"), [.3])
    np.testing.assert_allclose(area(out, "secdn"), [.4])
    assert selected(plan, "secnf_harv", SourceKey("secdn")) == .2
    assert not any(k.entry_origin in ("primn", "secdn") for k in out.keys)


@pytest.mark.parametrize("event", [
    conversion("retain_primary", "primf", "secdf", "retain"),
    conversion("create_primary", "secdf", "primf"),
    EventSpec("unresolved", "secdf", "pltns", "conversion", "unresolved", "keep", "proportional"),
])
def test_primary_prohibitions_and_unresolved_positive_event_are_atomic(event):
    s = state([("secdf", 1.5, "cleared", 1)], fixed={"primf": 1})
    before = s.snapshot()
    with pytest.raises(ValueError):
        run(s, [event], [[.1]])
    assert s.snapshot() == before


def test_primary_clear_has_no_finite_primary_age():
    s = state(fixed={"primf": 1})
    out, plan = run(s, [harvest("primf_harv", "proportional", "primf", "secdf")], [[.25]])
    np.testing.assert_allclose(area(out, "primf"), [.75])
    np.testing.assert_allclose(area(out, "secdf", 100.5, "cleared"), [.25])
    assert selected(plan, "primf_harv", SourceKey("primf")) == .25


def test_mixed_region_weights_close_and_preserve_region_specific_density():
    s = state([("secdf", 90.5, "c3ann", [1, 0]),
               ("secdf", 90.5, "cleared", [0, 2])], cells=2)
    weights = np.array([[.25, .75], [.6, .4]])
    out, plan = run(s, [harvest()], [[.4], [.5]], weights=weights)
    retained, cleared = SourceKey("secdf", 90.5, "c3ann"), SourceKey("secdf", 90.5, "cleared")
    np.testing.assert_allclose([[selected(plan, "secmf_harv", k, r) for k in (retained, cleared)]
                               for r in (0, 1)], [[.1, .3], [.3, .2]])
    np.testing.assert_allclose(plan.requested_area[:, 0], [.4, .5])
    np.testing.assert_allclose(plan.executed_area, plan.requested_area)
    np.testing.assert_allclose(plan.opening.area.sum(axis=1), [1.45, 1.55])
    np.testing.assert_allclose(plan.closing.area.sum(axis=1), [1.45, 1.55])
    np.testing.assert_allclose(out.area.sum(axis=1), [1, 2])
    # Region 0 densities: retained=10, cleared=2. Region 1: 20,4.
    carbon = [.1 * 10 + .3 * 2, .3 * 20 + .2 * 4]
    np.testing.assert_allclose(carbon, [1.6, 6.8])
    explicit = [.4 * .25 * 10 + .5 * .6 * 2, .4 * .75 * 20 + .5 * .4 * 4]
    np.testing.assert_allclose(carbon, explicit)


def test_empty_year_integer_cursor_and_geometry_snapshot_continuation():
    s = state(fixed={"primf": 1}, year=2020)
    events = [harvest("primf_harv", "proportional", "primf", "secdf")]
    full = s
    plans = []
    for d in (.2, 0., .1):
        full, plan = run(full, events, [[d]])
        plans.append(plan)
    partial, _ = run(s, events, [[.2]])
    partial, zero = run(partial, events, [[0.]])
    assert zero.next_year == 2022 and zero.event_time == 2021.5
    assert zero.selected.area.size == 0 and zero.opening.keys == zero.closing.keys
    np.testing.assert_array_equal(zero.opening.area, zero.closing.area)
    restored = GeometryState.from_snapshot(json.loads(json.dumps(partial.snapshot())))
    resumed, final_plan = run(restored, events, [[.1]])
    assert resumed.snapshot() == full.snapshot()
    assert [p.next_year for p in plans] == [2021, 2022, 2023]
    assert final_plan.interval_start == 2022 and final_plan.event_time == 2022.5
    assert final_plan.environment_year == final_plan.interval_end == 2023
    with pytest.raises(ValueError, match="cursor"):
        advance_geometry(resumed, events, [[0]], year=2022, region_ids=("R",), region_fraction=[[1]])
    with pytest.raises(ValueError, match="integer"):
        replace(restored, next_year=2022.5)
    with pytest.raises(ValueError, match="legacy"):
        GeometryState.from_snapshot(dict(restored.snapshot(), schema="land-events-v1"))


def test_key_and_event_permutations_are_equivalent_and_duplicate_keys_rejected():
    s = state([("secdf", 1.5, "cleared", .4), ("secdf", 90.5, "c3ann", .6)])
    reordered = replace(s, keys=s.keys[::-1], area=s.area[:, ::-1])
    assert reordered.snapshot() == s.snapshot()
    specs = [harvest("b"), conversion("a", "secdf", "c3ann")]
    a, ap = run(s, specs, [[.2, .1]])
    b, bp = run(reordered, specs[::-1], [[.1, .2]])
    assert a.snapshot() == b.snapshot()
    np.testing.assert_array_equal(ap.selected.area, bp.selected.area)
    with pytest.raises(ValueError, match="duplicate"):
        replace(s, keys=s.keys + (s.keys[-1],), area=np.column_stack((s.area, s.area[:, -1])))


def test_positive_tiny_area_is_retained_and_zero_source_differences_are_recorded():
    s = state([("secdf", 1.5, "cleared", 1e-250)], fixed={"primf": 1})
    out, plan = run(s, [harvest()], [[1e-251]])
    np.testing.assert_allclose(area(out, "secdf", 1.5, "cleared"), [1e-250], rtol=1e-15, atol=0)
    np.testing.assert_allclose(area(out, "secdf", 100.5, "cleared"), [0.], rtol=1e-15, atol=0)
    assert plan.selected.area[0] > 0
    empty = state(fixed={"primf": 1})
    cleared, diagnostic = run(empty, [harvest()], [[1e-250]])
    assert cleared.numerical_area_reduction[0] == 1e-250
    assert diagnostic.requested_area[0,0] == 1e-250
    assert diagnostic.executed_area[0,0] == 0
    with pytest.raises(AreaShortage):
        run(empty, [harvest()], [[1e-6]])


def test_tiny_keep_inflow_survives_complete_removal_of_existing_target():
    s = state([("secdf", 1.5, "c3ann", 1e-250), ("pltns", 1.5, "c3ann", 1)])
    out, _ = run(s, [conversion("keep", "secdf", "pltns", "retain", "keep"),
                    harvest("clear", "proportional", "pltns", "pltns")], [[1e-250, 1]])
    # Adding the inflow to opening and then subtracting 1 would erase it.
    np.testing.assert_array_equal(area(out, "pltns", 1.5, "c3ann"), [1e-250])
    np.testing.assert_array_equal(area(out, "pltns", 100.5, "cleared"), [1])


def test_initialization_requires_zero_finite_forest_and_region_weights_are_not_area():
    a = np.zeros((1, len(CLASSES)))
    a[0, 0] = 1
    s = GeometryState.from_class_area(CLASSES, [99], a, next_year=850, strategy="proportional")
    assert s.next_year == 850
    a[0, 2] = 1e-250
    with pytest.raises(ValueError, match="ancestry"):
        GeometryState.from_class_area(CLASSES, [99], a, next_year=850, strategy="proportional")
    with pytest.raises(ValueError, match="sum to one"):
        run(s, [], np.empty((1, 0)), weights=np.array([[.2, .3]]))


def test_zero_unresolved_event_remains_dormant_and_empty_batch_advances():
    s = state(fixed={"primf": 1})
    event = EventSpec("unresolved", "secdf", "pltns", "conversion", "unresolved", "keep", "proportional")
    out, plan = run(s, [event], [[0]])
    assert out.next_year == 101 and plan.selected.area.size == 0
    second, empty = run(out, [], np.empty((1, 0)))
    assert second.next_year == 102 and empty.requested_area.shape == (1, 0)


@pytest.mark.parametrize("cohorts", [128, 512])
def test_many_small_cohorts_preserve_area_and_half_harvest(cohorts):
    # An exact budget with a large primary area and many much smaller cohorts.
    # Pairwise summation must preserve the total-area and event-reservation
    # identities for both cohort counts.
    from math import fsum
    tiny = 1e-8
    s = state([("secdf", i + .5, "cleared", tiny) for i in range(cohorts)],
              fixed={"primf": 1.}, cells=2, year=1000)
    demand = cohorts * tiny / 2
    out, plan = run(s, [harvest(selection="proportional")], [[demand], [demand]])
    expected_total = fsum((1., cohorts * tiny))
    for row in out.area:
        assert abs(fsum(row) - expected_total) <= 16 * np.finfo(float).eps * expected_total
    np.testing.assert_allclose(plan.executed_area, [[2 * demand]], rtol=16*np.finfo(float).eps, atol=0)
    # Half of each cohort is selected, but that area stays on its birth year.
    for i in range(cohorts):
        np.testing.assert_allclose(area(out, "secdf", i + .5, "cleared"),
                                   tiny, rtol=16*np.finfo(float).eps, atol=0)
    np.testing.assert_allclose(area(out, "secdf", 1000.5, "cleared"),
                               0., rtol=16*np.finfo(float).eps, atol=0)

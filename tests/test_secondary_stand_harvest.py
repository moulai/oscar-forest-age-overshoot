"""Secondary wood harvest keeps the stand; new forest still opens a cohort."""
import numpy as np
import pytest

from oscar._core._model._modules.land_geometry import EventSpec, LandClass
from oscar._core._model._modules.regional_forest_age import (
    RegionalLayout, apply_annual_events, total_carbon)
from oscar._core._model._modules.regional_forest_secondary_carbon import (
    PROJECTION, apply_secondary_agb_request_events)


CLASSES = (
    LandClass("primf", "Forest", "forest_unaged"),
    LandClass("secdf", "Forest", "forest_cohort"),
    LandClass("crop", "Crop", "nonforest"),
)
EVENTS = (
    EventSpec("secmf_harv", "secdf", "secdf", "harvest", "clear", "reset", "proportional"),
    EventSpec("secyf_harv", "secdf", "secdf", "harvest", "clear", "reset", "proportional"),
    EventSpec("primf_harv", "primf", "secdf", "harvest", "clear", "reset", "proportional"),
    EventSpec("clear_establish", "crop", "secdf", "conversion", "clear", "reset", "proportional"),
    EventSpec("regrowth", "crop", "secdf", "conversion", "retain", "reset", "proportional", "source_p_wood"),
)
BIRTHS = (1.5, 20.5, 30.5)


def layout():
    return RegionalLayout(("R",), ("m",), CLASSES, BIRTHS, ("Forest", "Crop"), ("short", "long"), EVENTS)


def state(model):
    out = model.empty_state(30.5)
    out.rf_A_primary.values[:] = 20
    out.rf_A_forest.values[:] = [[[30., 10., 0.]]]
    out.rf_A_nonforest.values[:] = 40
    out.rf_W_primary.values[:] = [[2.]]
    out.rf_F_primary.values[:] = [[.2]]
    out.rf_W_forest.values[:] = [[[[3., .2, 0.]]]]
    out.rf_F_forest.values[:] = [[[[.3, .1, 0.]]]]
    out.rf_C_nonforest.values[:] = [[[.8]]]
    out.rf_C_cwd.values[:] = [[[.4, .2]]]
    out.rf_C_soil.values[:] = [[[4., 2.]]]
    return out


def routing():
    pwood = np.array([[[.8, .25]]])
    phwp = np.broadcast_to([.2, .3], (1, 1, 2, 2)).copy()
    weights = np.ones((1, len(EVENTS), len(layout().slots)))
    return pwood, phwp, weights


def test_secondary_harvest_removes_carbon_without_resetting_age():
    model = layout()
    before = state(model)
    pwood, phwp, weights = routing()
    # Four hectares, one tenth of each cohort. The clear-cut fraction of stock.
    demand = np.array([[4., 0., 0., 0., 0.]])
    after, ledger = apply_annual_events(before, model, 30.5, demand, weights, pwood, phwp)
    fraction = np.array([.1, .1, 0.])
    wood = before.rf_W_forest.values[0, 0, 0]
    other = before.rf_F_forest.values[0, 0, 0]
    removed_wood = (wood * fraction).sum()
    removed_other = (other * fraction).sum()
    np.testing.assert_allclose(ledger.removed_wood.values[0, 0].sum(), removed_wood)
    np.testing.assert_allclose(ledger.removed_other.values[0, 0].sum(), removed_other)
    np.testing.assert_allclose(after.rf_W_forest.values[0, 0, 0], wood * (1. - fraction))
    np.testing.assert_allclose(after.rf_F_forest.values[0, 0, 0], other * (1. - fraction))
    np.testing.assert_allclose(after.rf_A_forest.values[0, 0, 0], [30., 10., 0.])
    np.testing.assert_array_equal(after.birth_time, before.birth_time)
    np.testing.assert_allclose(after.rf_A_forest.sum(), before.rf_A_forest.sum())
    np.testing.assert_allclose(total_carbon(after), total_carbon(before))
    np.testing.assert_allclose(ledger.executed_area, demand)
    np.testing.assert_allclose(ledger.hwp_added.sum() + ledger.cwd_added.sum(), removed_wood)
    np.testing.assert_allclose(ledger.soil_added.sum(), removed_other)


def test_request_q_keeps_the_same_removal_and_the_same_area():
    model = layout()
    before = state(model)
    # Eligible cohorts only: the two stands already alive at the event.
    slots = model.slots
    cols = [i for i, (name, birth) in enumerate(slots) if name == "secdf" and birth < 30.5]
    weights = np.zeros((1, len(EVENTS), len(slots)))
    weights[:, 0, cols] = 1.
    weights[:, 1, cols] = 1.
    pwood, phwp, _ = routing()
    # G = W + 0.5 F on 40 ha. Q takes one quarter of that gross carbon.
    wood = before.rf_W_forest.values[0, 0, 0, :2]
    other = before.rf_F_forest.values[0, 0, 0, :2]
    area = before.rf_A_forest.values[0, 0, 0, :2]
    gross = (wood + .5 * other).sum()
    request = np.zeros((1, len(EVENTS)))
    after, ledger = apply_secondary_agb_request_events(
        before, model, 30.5, request, weights, pwood, phwp, np.array([gross / 4.]),
        np.array([[gross / 8., gross / 8.]]), np.array([[.5]]),
        secondary_events=("secmf_harv", "secyf_harv"), projection=PROJECTION,
        k_f_ag_leaf_only=np.array([[.2]]))
    np.testing.assert_allclose(ledger.sec_agb_achieved, gross / 4.)
    np.testing.assert_allclose(ledger.sec_agb_area, area.sum() / 4.)
    np.testing.assert_allclose(after.rf_A_forest.values[0, 0, 0], [30., 10., 0.])
    np.testing.assert_array_equal(after.birth_time, before.birth_time)
    np.testing.assert_allclose(after.rf_W_forest.values[0, 0, 0, :2], wood * .75)
    np.testing.assert_allclose(after.rf_F_forest.values[0, 0, 0, :2], other * .75)
    np.testing.assert_allclose(total_carbon(after), total_carbon(before))
    np.testing.assert_allclose(ledger.requested_area[:, :2].sum(), ledger.executed_area[:, :2].sum())


def test_primary_harvest_and_clear_establishment_open_a_zero_carbon_cohort():
    model = layout()
    before = state(model)
    pwood, phwp, weights = routing()
    demand = np.array([[0., 0., 5., 4., 0.]])
    after, ledger = apply_annual_events(before, model, 30.5, demand, weights, pwood, phwp)
    np.testing.assert_allclose(after.rf_A_primary, 15.)
    np.testing.assert_allclose(after.rf_A_forest.values[0, 0, 0], [30., 10., 9.])
    np.testing.assert_allclose(after.rf_W_forest.values[0, 0, 0, -1], 0.)
    np.testing.assert_allclose(after.rf_F_forest.values[0, 0, 0, -1], 0.)
    np.testing.assert_allclose(after.birth_time[-1], 30.5)
    np.testing.assert_allclose(ledger.removed_wood.values[0, 0, 2], .5)
    np.testing.assert_allclose(total_carbon(after), total_carbon(before))
    # The cleared crop carbon moved once; it is not also live on the new cohort.
    np.testing.assert_allclose(ledger.removed_wood.values[0, 0, 3] + ledger.removed_other.values[0, 0, 3], .08)


def test_retained_establishment_still_opens_at_the_event_year():
    """Prepared non-forest to secdf keeps the donor split; it is not a clear-cut."""
    model = layout()
    before = state(model)
    pwood, phwp, weights = routing()
    demand = np.array([[0., 0., 0., 0., 4.]])
    after, _ = apply_annual_events(before, model, 30.5, demand, weights, pwood, phwp)
    np.testing.assert_allclose(after.rf_A_forest.values[0, 0, 0], [30., 10., 4.])
    np.testing.assert_allclose(after.birth_time[-1], 30.5)
    # 4/40 of crop carbon 0.8, split by donor p_wood 0.25.
    np.testing.assert_allclose(after.rf_W_forest.values[0, 0, 0, -1], .02)
    np.testing.assert_allclose(after.rf_F_forest.values[0, 0, 0, -1], .06)
    np.testing.assert_allclose(total_carbon(after), total_carbon(before))

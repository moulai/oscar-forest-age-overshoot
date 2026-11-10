"""Small, inspectable physical examples for regional annual cohorts."""
from dataclasses import replace
import unittest
import json
from pathlib import Path
import sys
import numpy as np
import xarray as xr

from oscar._core._model._modules.land_geometry import LandClass, EventSpec
from oscar._core._model._modules.dead_carbon import DeadCarbonRates
from oscar._core._model._modules.regional_forest_age_allocation import select_area, select_joint_area
from oscar._core._model._modules.regional_forest_age import (
    RegionalLayout, RegionalRates, apply_annual_events as apply_member_events, advance_carbon, total_carbon)


def apply_annual_events(state, layout, time, demand, weights, pwood, phwp):
    """Independent member calls for the analytical shared-geometry examples.

    The production API reserves the batched event boundary to native Model.
    These tests check analytically specified selections and transfers.
    """
    states, ledgers = [], []
    for mi, member in enumerate(layout.members):
        single = replace(layout, members=(member,), source_members=(layout.parameter_ids[mi],)
                         if layout.source_members is not None else None)
        after, ledger = apply_member_events(state.isel(config=slice(mi, mi+1)), single, time,
            demand, weights, pwood[mi:mi+1], phwp[mi:mi+1])
        states.append(after); ledgers.append(ledger)
    return tuple(xr.concat(items, dim="config", data_vars="minimal", coords="minimal", compat="equals", join="exact")
                 for items in (states, ledgers))


def fixture():
    classes = (LandClass("primf", "Forest", "forest_unaged"),
               LandClass("secdf", "Forest", "forest_cohort"),
               LandClass("crop", "Crop", "nonforest"))
    events = (
        EventSpec("conversion", "secdf", "crop", "conversion", "clear", "none", "proportional"),
        EventSpec("regrowth", "crop", "secdf", "conversion", "retain", "reset", "proportional", "source_p_wood"),
        EventSpec("primary_harvest", "primf", "secdf", "harvest", "clear", "reset", "proportional"),
        EventSpec("mature_harvest", "secdf", "secdf", "harvest", "clear", "reset", "oldest"),
        EventSpec("young_harvest", "secdf", "secdf", "harvest", "clear", "reset", "youngest"))
    layout = RegionalLayout(("R1",), ("member-A", "member-B"), classes,
                            (.5, 5.5, 10.5), ("Forest", "Crop"), ("short", "long"), events)
    state = layout.empty_state(10.5)
    state.rf_A_primary.values[:] = 20
    state.rf_A_forest.values[:] = [[[30., 10., 0.]]]
    state.rf_A_nonforest.values[:] = 40
    state.rf_W_primary.values[:] = [[2.], [4.]]
    state.rf_F_primary.values[:] = [[.2], [.4]]
    state.rf_W_forest.values[:] = [[[[3., .2, 0.]]], [[[6., .4, 0.]]]]
    state.rf_F_forest.values[:] = [[[[.3, .1, 0.]]], [[[.6, .2, 0.]]]]
    state.rf_C_nonforest.values[:] = [[[.8]], [[1.6]]]
    state.rf_C_cwd.values[:] = [[[.4, .2]], [[.8, .4]]]
    state.rf_C_soil.values[:] = [[[4., 2.]], [[8., 4.]]]
    pwood = np.array([[[.8, .25]], [[.8, .25]]])
    phwp = np.broadcast_to([.2, .3], (2, 1, 2, 2)).copy()
    weights = np.ones((1, 5, len(layout.slots)))
    return layout, state, pwood, phwp, weights


class AllocationTests(unittest.TestCase):
    def test_preference_and_supply_limitation(self):
        for demand, expected in [(10, [5, 5]), (80, [10, 70]), (100, [10, 90])]:
            h, gap = select_area([10, 90], [9, 1], demand)
            np.testing.assert_allclose(h, expected)
            self.assertEqual(gap, 0.)

    def test_eligibility_and_actual_wood_target(self):
        h, gap = select_area([10, 90], [1, 0], 12)
        np.testing.assert_allclose(h, [10, 0])
        self.assertEqual(gap, 2.)
        h, gap = select_area([10, 90], [1, 1], 20, yield_density=[0, .5])
        np.testing.assert_allclose(h, [0, 40])
        self.assertEqual(gap, 0.)

    def test_joint_permutation_and_shared_supply(self):
        a, w, d = np.array([10., 90.]), np.array([[9., 1.], [1., 4.]]), np.array([30., 40.])
        h = select_joint_area(a, w, d)
        np.testing.assert_allclose(h.sum(axis=1), d, atol=1e-8)
        self.assertTrue(np.all(h.sum(axis=0) <= a + 1e-10))
        np.testing.assert_allclose(h, select_joint_area(a, w[::-1], d[::-1])[::-1], atol=1e-8)
        np.testing.assert_allclose(h[:, ::-1], select_joint_area(a[::-1], w[:, ::-1], d), atol=1e-8)

    def test_eligibility_subset_failure_and_boundary(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            select_joint_area([2., 8.], [[1., 0.], [1., 1.]], [3., 2.])
        h = select_joint_area([2., 8.], [[1., 0.], [1., 1.]], [2., 2.])
        np.testing.assert_allclose(h, [[2., 0.], [0., 2.]], atol=1e-8)


class CarbonTests(unittest.TestCase):
    def test_actual_density_and_product_routing(self):
        layout, state, pwood, phwp, weights = fixture()
        demand = np.array([[0., 0., 0., 4., 0.]])
        out, diag = apply_annual_events(state, layout, 10.5, demand, weights, pwood, phwp)
        # Proportional selection takes 3 ha-equivalent of old and 1 of young.
        np.testing.assert_allclose(diag.selected_area.values[0, 3], [0, 3, 1, 0, 0])
        np.testing.assert_allclose(diag.removed_wood.values[:, 0, 3], [.32, .64])
        np.testing.assert_allclose(diag.hwp_added.values[:, 0, 3].sum(axis=-1), [.16, .32])
        np.testing.assert_allclose(diag.cwd_added.values[:, 0, 3], [.16, .32])
        np.testing.assert_allclose(total_carbon(out), total_carbon(state))
        # The same fraction of stock leaves, but the stand is not reborn.
        np.testing.assert_allclose(out.rf_A_forest.values[0, 0, 0], [30, 10, 0])
        np.testing.assert_array_equal(out.birth_time, state.birth_time)
        np.testing.assert_allclose(out.rf_W_forest.values[..., -1], 0.)

    def test_primary_clearance_and_retained_establishment_differ(self):
        layout, state, pwood, phwp, weights = fixture()
        out, diag = apply_annual_events(state, layout, 10.5, [[0, 4, 5, 0, 0]], weights, pwood, phwp)
        self.assertEqual(out.rf_A_primary.isel(config=0).item(), 15.)
        self.assertEqual(out.rf_A_forest.values[0, 0, 0, -1], 9.)
        # The primary-harvest part contributes zero live carbon; the retained
        # crop part contributes 4/40 of crop C, split using donor p_wood=.25.
        np.testing.assert_allclose(out.rf_W_forest.values[:, 0, 0, -1], [.02, .04])
        np.testing.assert_allclose(out.rf_F_forest.values[:, 0, 0, -1], [.06, .12])
        np.testing.assert_allclose(diag.removed_wood.values[:, 0, 2], [.5, 1.])
        np.testing.assert_allclose(total_carbon(out), total_carbon(state))

    def test_conversions_reserve_supply_without_reusing_inflow(self):
        layout, state, pwood, phwp, weights = fixture()
        with self.assertRaisesRegex(ValueError, "exceed"):
            apply_annual_events(state, layout, 10.5, [[30, 20, 0, 20, 0]], weights, pwood, phwp)
        out, diag = apply_annual_events(state, layout, 10.5, [[30, 20, 0, 10, 0]], weights, pwood, phwp)
        np.testing.assert_allclose(diag.selected_area.sum("source_slot"), [[30, 20, 0, 10, 0]])
        np.testing.assert_allclose(total_carbon(out), total_carbon(state))

    def test_live_dead_exchange_and_clock(self):
        layout, state, *_ = fixture()
        shape = (2, 1, 3)
        production = np.full(shape, .01)
        mortality = np.full(shape, .03)
        fire = np.full(shape, .01)
        zeros = np.zeros(shape)
        route = np.broadcast_to([1., 0., .2], shape)
        dead = DeadCarbonRates(layout.members, layout.biomes, layout.products,
                               np.full((2, 2), .1), np.full((2, 2), .02),
                               np.full((2, 2), .4), np.full((2, 2, 2), .05))
        rates = RegionalRates(production, mortality, fire, zeros, zeros, route, (dead,))
        out, flux = advance_carbon(state, layout, 10.5, 11., rates)
        np.testing.assert_allclose(total_carbon(out)-total_carbon(state), flux.net_uptake.sum(("reg_land", "bio_land")), atol=1e-13)
        np.testing.assert_array_equal(out.birth_time, state.birth_time)
        np.testing.assert_array_equal(out.rf_A_forest, state.rf_A_forest)
        self.assertEqual(out.year.item(), 11.)
        self.assertGreater(out.rf_W_forest.values[0, 0, 0, 0], 0.)


def native_fixture(feedback=False):
    from oscar._core._model._modules.regional_forest_age_native import build_regional_land_model
    from oscar._core._model._modules.OSCAR_CO2 import OSCAR_CO2
    from oscar._core._params_spec.Par_CO2 import get_params
    layout, state, pwood, phwp, _ = fixture()
    layout = RegionalLayout(layout.regions, layout.members, layout.classes,
                            (.5, 5.5, 10.5, 11.5), layout.biomes, layout.products, layout.events)
    initial = layout.empty_state(10.)
    for name in state:
        if "birth_time" in state[name].dims:
            initial[name].loc[dict(birth_time=state.birth_time)] = state[name].drop_vars("year")
        else:
            initial[name] = state[name].drop_vars("year")
    initial["D_CO2"] = xr.zeros_like(initial.rf_W_primary.sum("reg_land"))
    dead = DeadCarbonRates(layout.members, layout.biomes, layout.products,
                           np.full((2, 2), .1), np.full((2, 2), .02),
                           np.full((2, 2), .4), np.full((2, 2, 2), .05))
    def rates(V, P, F, t0, t1):
        shape = (2, 1, 3)
        production = np.full(shape, .001)
        if feedback:
            co2, _ = xr.broadcast(V.D_CO2, V.rf_W_primary.sum("reg_land"))
            production *= np.exp(.1 * co2.transpose("config").values[:, None, None])
        return RegionalRates(production, np.full(shape, .03), np.full(shape, .01),
                             np.zeros(shape), np.zeros(shape),
                             np.broadcast_to([1., 0., .2], shape), (dead,))
    def reference(state, V, P, F, t0, t1):
        return xr.zeros_like(state.rf_W_primary)
    land = build_regional_land_model(layout, rates, reference,
                                     inputs=("event_area", "D_CO2"))
    model = land.merge(OSCAR_CO2.copy(only=["D_Fsink_CO2", "d_CO2", "D_CO2", "Eimb_CO2"]))
    parameters = xr.Dataset({
        "rf_p_wood": (("config", "reg_land", "bio_land"), pwood),
        "rf_p_hwp": (("config", "reg_land", "bio_land", "box_hwp"), phwp),
        "rf_NBP_pi": xr.zeros_like(initial.rf_W_primary),
        "a_CO2": get_params().a_CO2,
    }, coords={key: initial[key] for key in ("config", "reg_land", "bio_land", "box_hwp")})
    forcing = xr.Dataset({name: ("year", [0., 0., 0.]) for name in
                          ["D_Eff", "D_Foxi_CH4", "D_Focean", "D_Epf_CO2", "D_Emiss_CO2"]},
                         coords={"year": [10, 11, 12], "reg_land": list(layout.regions),
                                 "event": [ev.event_id for ev in layout.events]})
    forcing["event_area"] = (("year", "reg_land", "event"), [[[0., 0., 0., 0., 0.]],
                                                             [[0., 0., 0., 4., 0.]],
                                                             [[0., 0., 0., 4., 0.]]])
    return model, initial, parameters, forcing


def run_native(model, initial, parameters, forcing, nt=3):
    return model(initial, parameters, forcing, dtype=float, nt=nt, adapt_nt=None,
                 get_final=True, var_keep=["D_Eluc", "D_Fland", "rf_event_count", "rf_NBP",
                                           "rf_NPP", "rf_HWP_input", "Eimb_CO2"])


class NativeTests(unittest.TestCase):
    def test_native_co2_receives_actual_budget_and_events_once(self):
        for nt in (1, 3):
            model, initial, parameters, forcing = native_fixture()
            output, final = run_native(model, initial, parameters, forcing, nt)
            np.testing.assert_allclose(total_carbon(final) + parameters.a_CO2 * final.D_CO2,
                                       total_carbon(initial), atol=1e-12)
            np.testing.assert_allclose(output.rf_event_count, [0., 1., 1.])
            self.assertGreater(output.rf_HWP_input.sel(year=11).min().item(), 0.)
            self.assertGreater(final.D_CO2.min().item(), 0.)
            # The prescribed 8 Mha is allocated from the surviving source in
            # two annual events, not repeated for every numerical substep.
            np.testing.assert_allclose(final.rf_A_forest.sum("birth_time"), initial.rf_A_forest.sum("birth_time"))
            self.assertEqual(final.rf_A_forest.sel(config="member-A", birth_time=11.5).item(), 0.)
            np.testing.assert_allclose(final.rf_A_forest.sel(config="member-A", birth_time=.5).item(), 30.)
            np.testing.assert_allclose(final.rf_A_forest.sel(config="member-A", birth_time=5.5).item(), 10.)

    def test_native_restart_and_prescribed_co2(self):
        model, initial, parameters, forcing = native_fixture(feedback=True)
        full, final = run_native(model, initial, parameters, forcing)
        _, middle = run_native(model, initial, parameters, forcing.sel(year=[10, 11]))
        resumed, resumed_final = run_native(model, middle, parameters, forcing.sel(year=[11, 12]))
        xr.testing.assert_allclose(resumed_final, final)
        xr.testing.assert_allclose(resumed.sel(year=12), full.sel(year=12))
        fixed = forcing.assign(D_CO2=("year", [0., 2., 2.]), d_CO2=("year", [0., 0., 0.]))
        prescribed, _ = run_native(model, initial, parameters, fixed)
        self.assertGreater(prescribed.rf_NPP.sel(year=11).sum().item(), full.rf_NPP.sel(year=11).sum().item())
        np.testing.assert_allclose(prescribed.Eimb_CO2, -prescribed.D_Eluc.sum("reg_land"), atol=1e-12)


def write_diagnostics(directory):
    """Export the same small physical cases for inspection and figure inputs."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    cases = []
    for demand in (10., 80., 100.):
        selected, missing = select_area([10., 90.], [9., 1.], demand)
        cases.append(dict(requested_area=demand, area=[10., 90.], weights=[9., 1.],
                          selected_area=selected.tolist(), unmet_area=missing))
    layout, initial, pwood, phwp, weights = fixture()
    after, transfers = apply_annual_events(initial, layout, 10.5, [[0., 4., 5., 4., 0.]], weights, pwood, phwp)
    initial.to_netcdf(path / "initial_state.nc")
    after.to_netcdf(path / "after_events.nc")
    transfers.to_netcdf(path / "event_transfers.nc")
    summary = dict(scope="synthetic physical-kernel verification; not a calibrated historical simulation",
                   units=dict(area="Mha", carbon="PgC", time="yr"),
                   preference_cases=cases,
                   initial_carbon=total_carbon(initial).values.tolist(),
                   final_carbon=total_carbon(after).values.tolist(),
                   event_budget_residual=transfers.budget_residual.values.tolist(),
                   direct_event_atmosphere=0.,
                   explanation="These clearances route all removed carbon to products or dead pools. Subsequent decay emits carbon; removal is not immediate emission.")
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def write_native_diagnostics(directory):
    """Save a native-land/CO2 run whose complete budget can be inspected."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model, initial, parameters, forcing = native_fixture(feedback=True)
    output, final = run_native(model, initial, parameters, forcing)
    initial.to_netcdf(path / "initial_state.nc")
    output.to_netcdf(path / "native_output.nc")
    final.to_netcdf(path / "final_state.nc")
    forcing.to_netcdf(path / "forcing.nc")
    atmospheric_gain = parameters.a_CO2 * final.D_CO2
    residual = total_carbon(final) + atmospheric_gain - total_carbon(initial)
    summary = dict(scope="synthetic native regional-land and OSCAR_CO2 integration; not full OSCAR or historical validation",
                   units=dict(carbon="PgC", co2="ppm", area="Mha", time="yr"),
                   initial_land_carbon=total_carbon(initial).values.tolist(),
                   final_land_carbon=total_carbon(final).values.tolist(),
                   atmospheric_carbon_gain=atmospheric_gain.values.tolist(),
                   final_co2_anomaly=final.D_CO2.values.tolist(),
                   total_budget_residual=residual.values.tolist(),
                   annual_event_count=output.rf_event_count.values.tolist(),
                   explanation="Actual live/dead/product stocks exchange carbon with the ordinary OSCAR CO2 component in the same interval. The synthetic reference sink and PI net uptake are both zero.")
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--diagnostics":
        write_diagnostics(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "--native-diagnostics":
        write_native_diagnostics(sys.argv[2])
    else:
        unittest.main()

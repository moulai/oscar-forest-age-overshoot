"""Independent cell x region carbon oracle for the compressed geometry route.

The expected route uses singleton-cell geometry only for selected area. It
keeps local W/F stocks and frozen regional nonforest/dead pools. It never calls
apply_events, never derives forest removal from regional mean density, and
uses an independent analytic constant-coefficient live solution. Synthetic
members are deliberately unrelated to the production parameter ensemble.
"""
from dataclasses import dataclass, replace
import copy
import numpy as np
import pytest

from oscar._core._model._modules.land_geometry import (
    LandClass, EventSpec, SourceKey, GeometryState, advance_geometry, AreaTable)
from oscar._core._model._modules.forest_age import TwoPoolForestState
from oscar._core._model._modules.land_events import LandEventState, EventRouting, apply_events
from oscar._core._model._modules.land_carbon import (
    LandCarbonCoefficients, AgeProduction, advance_land_carbon)
from oscar._core._model._modules.dead_carbon import DeadCarbonRates, advance_dead_carbon
from oscar._core._model._modules.dynamic_reference import BIOMES
from oscar._core._model._modules.growth_curve import GrowthCurve

MEMBERS = (901, 107)  # Synthetic identities; order is intentional.
REGIONS = ("synthetic_west", "synthetic_east")
PRODUCTS = ("short", "long")
WEIGHTS = np.array([[.25, .75], [1., 0.], [0., 1.]])
CLASSES = (
    LandClass("primf", "Forest", "forest_unaged"),
    LandClass("secdf", "Forest", "forest_cohort"),
    LandClass("pltnf", "Forest", "forest_cohort"),
    LandClass("range", "Non-Forest", "nonforest"),
    LandClass("secdn", "Non-Forest", "nonforest"),
    LandClass("c3ann", "Cropland", "nonforest"),
    LandClass("pastr", "Pasture", "nonforest"),
    LandClass("urban", "Urban", "nonforest"),
)
CLASS = {c.class_id: c for c in CLASSES}
NF = tuple(c.class_id for c in CLASSES if c.role == "nonforest")
EVENTS = (
    EventSpec("keep_existing", "secdf", "pltnf", "conversion", "retain", "keep", "proportional"),
    EventSpec("old_clear", "secdf", "secdf", "harvest", "clear", "reset", "oldest"),
    EventSpec("young_clear", "secdf", "secdf", "harvest", "clear", "reset", "youngest"),
    EventSpec("forest_exit", "secdf", "range", "conversion", "retain", "none", "proportional"),
    EventSpec("range_reentry", "range", "secdf", "conversion", "retain", "reset", "proportional", "source_p_wood"),
    EventSpec("crop_clear_entry", "c3ann", "secdf", "conversion", "clear", "reset", "proportional"),
    EventSpec("nonforest_harvest", "range", "range", "harvest", "clear", "none", "proportional"),
    EventSpec("nonforest_conversion", "range", "c3ann", "conversion", "clear", "none", "proportional"),
    EventSpec("primary_clear", "primf", "secdf", "harvest", "clear", "reset", "proportional"),
)


class Checks:
    """Sixteen float64 eps at the participating quantity scale, no floor."""
    def __init__(self):
        self.count = 0
        self.worst = {}

    def equal(self, name, actual, expected, scale=None):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert np.isfinite(a).all() and np.isfinite(b).all(), name
        magnitude = np.maximum(abs(a), abs(b))
        if scale is not None:
            magnitude = np.maximum(magnitude, np.broadcast_to(scale, a.shape))
        error = abs(a-b)
        tolerance = 16*np.finfo(float).eps*magnitude
        assert np.all(error <= tolerance), (name, float(error.max()), float(tolerance.max()), a, b)
        eps_error = np.divide(error, np.finfo(float).eps*magnitude,
                              out=np.zeros_like(error), where=magnitude > 0)
        self.worst[name] = max(self.worst.get(name, 0.), float(eps_error.max(initial=0.)))
        self.count += a.size


def analytic_live(stock, production, mortality, fire, dt, crop=0., grazing=0.):
    """Solve dC/dt=P_eff-kC with long-double arithmetic, independently."""
    c, p, mu, fi, cr, gr = np.broadcast_arrays(*[
        np.asarray(v, np.longdouble) for v in (stock, production, mortality, fire, crop, grazing)])
    k = mu+fi
    assert np.all(k > 0)  # This bounded experiment has no degenerate rates.
    survival = np.exp(-k*np.longdouble(dt))
    effective = p*(1-cr-gr)
    end = c*survival + effective*(-np.expm1(-k*dt))/k
    integrated = (c + effective*dt - end)/k
    return {"carbon": np.asarray(end, float), "production": np.asarray(p*dt, float),
            "mortality": np.asarray(mu*integrated, float), "fire": np.asarray(fi*integrated, float),
            "crop_harvest": np.asarray(cr*p*dt, float), "grazing": np.asarray(gr*p*dt, float)}


def initial_geometry(strategy):
    keys = (SourceKey("primf"), SourceKey("secdf", 1985., "cleared"),
            SourceKey("secdf", 1990., "cleared"), SourceKey("secdf", 1990., "range"),
            SourceKey("pltnf", 1990., "range"), *(SourceKey(n) for n in NF))
    area = np.array([[2., 1., 1., 1., .5, 2., 1., 1., 1., 1.],
                     [2., .5, 1.5, .5, .5, 2., 1., 1., 1., 1.],
                     [2., 1.5, .5, 1., .5, 2., 1., 1., 1., 1.]])
    return GeometryState(CLASSES, (11, 22, 33), keys, area, 2000, strategy)


def routing():
    wood = np.array([[.7, .6, .4, .3, .1], [.65, .5, .35, .25, .15]])
    products = np.array([[[.2, .3], [.1, .25], [.15, .2], [.05, .2], [0., 0.]],
                         [[.15, .25], [.2, .1], [.1, .1], [.1, .15], [0., 0.]]])
    return EventRouting(MEMBERS, BIOMES, PRODUCTS, wood, products)


def coefficients(ri):
    member = np.array([1., 1.4])[:, None]
    production = member*np.array([[.3, .2, .25, .2, .15, 0.]])*(1+ri)
    mortality = member*np.array([[.08, .2, .1, .12, .09, .01]])
    fire = member*np.array([[.02, .03, .015, .025, .01, .01]])
    crop = np.tile([0., 0., .03, .2, 0., 0.], (2, 1))
    grazing = np.tile([0., 0., .04, 0., .15, 0.], (2, 1))
    route = np.tile([1., 0., .3, .15, .2, 0.], (2, 1))
    dead = DeadCarbonRates(MEMBERS, BIOMES, PRODUCTS,
        np.full((2, 5), .12), np.full((2, 5), .04), np.full((2, 5), .35),
        np.broadcast_to([.4, .08], (2, 5, 2)))
    return LandCarbonCoefficients(MEMBERS, REGIONS[ri], production, mortality,
                                  fire, crop, grazing, route, dead)


def age_setting(ri):
    slopes = np.array([.02, .015])
    curve = GrowthCurve(REGIONS, "area", np.array([0., 20.]),
        np.array([[[0., s, 0., 0.]] for s in slopes]), 1-20*slopes, slopes/(1-20*slopes))
    return AgeProduction(curve, REGIONS[ri], MEMBERS, np.array([.2, .3]), tau_years=2.)


@dataclass
class LocalOracle:
    """Only forest is local; nonforest and dead pools are regional mixtures."""
    geometry: GeometryState
    local: list  # region -> cell -> {complete key: actual member x W/F stock}
    nonforest: list
    cwd: list
    soil: list
    hwp: list
    time: float = 2000.

    @classmethod
    def start(cls, geometry):
        local, nf, cwd, soil, hwp = [], [], [], [], []
        for ri in range(2):
            cells = []
            for ci in range(3):
                stocks = {}
                for ki, key in enumerate(geometry.keys):
                    if CLASS[key.current_class].role == "nonforest":
                        continue
                    density = ([20., 5.] if key.birth_time is None else
                               [9., 2.] if key.birth_time == 1985. else
                               [12., 4.] if key.entry_origin == "range" else [2., 1.])
                    stocks[key] = np.array(density)[None, :]*np.array([1., 1.5])[:, None]*(1+2*ri)*geometry.area[ci, ki]*WEIGHTS[ci, ri]
                cells.append(stocks)
            local.append(cells)
            areas = {key: np.dot(geometry.area[:, ki], WEIGHTS[:, ri]) for ki, key in enumerate(geometry.keys)}
            nf.append(np.column_stack([np.array([1., 1.5])*4*(1+2*ri)*(j+1)*areas[SourceKey(n)]
                                        if n != "urban" else np.zeros(2) for j, n in enumerate(NF)]))
            biome_area = np.array([sum(a for key, a in areas.items() if CLASS[key.current_class].biome == b) for b in BIOMES])
            cwd.append(np.array([1., 1.5])[:, None]*np.array([3., 2., 1., .5, 0.])[None, :]*biome_area)
            soil.append(np.array([1., 1.5])[:, None]*np.array([15., 10., 8., 6., 0.])[None, :]*biome_area)
            hwp.append(np.zeros((2, 5, 2)))
        return cls(geometry, local, nf, cwd, soil, hwp)

    def area(self, ri, key):
        if key not in self.geometry.keys:
            return 0.
        return np.dot(self.geometry.area[:, self.geometry.keys.index(key)], WEIGHTS[:, ri])

    def aggregate(self, ri):
        keys = [k for k in self.geometry.keys if CLASS[k.current_class].role == "forest_cohort"]
        primary = sum((cell.get(SourceKey("primf"), np.zeros((2, 2))) for cell in self.local[ri]), np.zeros((2, 2)))
        carbon = np.stack([sum((cell.get(k, np.zeros((2, 2))) for cell in self.local[ri]), np.zeros((2, 2))) for k in keys], axis=2)
        forest = TwoPoolForestState(self.time, MEMBERS, self.area(ri, SourceKey("primf")), primary,
            [k.birth_time for k in keys], [self.area(ri, k) for k in keys], carbon, REGIONS[ri], tuple(k.entry_origin for k in keys))
        return LandEventState(forest, CLASSES, BIOMES, PRODUCTS, tuple(k.current_class for k in keys),
            [self.area(ri, SourceKey(n)) for n in NF], self.nonforest[ri], self.cwd[ri], self.soil[ri], self.hwp[ri], self.geometry.next_year)

    def continuous(self, coeff, age_on, dt):
        ledgers = []
        for ri, co in enumerate(coeff):
            ledger = {n: np.zeros((2, 5)) for n in ("production", "fire", "crop_harvest", "grazing", "live_to_cwd", "live_to_soil")}
            for ci, cell in enumerate(self.local[ri]):
                for key, stock in tuple(cell.items()):
                    area = self.geometry.area[ci, self.geometry.keys.index(key)]*WEIGHTS[ci, ri]
                    factors = np.ones((2, 2))
                    if age_on and key.birth_time is not None:
                        age = self.time+dt/2-key.birth_time
                        assert 0 <= age < 20
                        slope = (.02, .015)[ri]
                        factors[:, 0] = slope*age+slope/np.array([.2, .3])
                        factors[:, 1] = -np.expm1(-age/2.)
                    step = analytic_live(stock, co.live_production[:, :2]*factors*area,
                                         co.live_mortality[:, :2], co.live_fire[:, :2], dt)
                    cell[key] = step["carbon"]
                    for field in ("production", "fire", "crop_harvest", "grazing"):
                        ledger[field][:, 0] += step[field].sum(axis=1)
                    ledger["live_to_cwd"][:, 0] += step["mortality"][:, 0]
                    ledger["live_to_soil"][:, 0] += step["mortality"][:, 1]
            for j, name in enumerate(NF):
                bi = BIOMES.index(CLASS[name].biome)
                pi = bi+1
                step = analytic_live(self.nonforest[ri][:, j], co.live_production[:, pi]*self.area(ri, SourceKey(name)),
                    co.live_mortality[:, pi], co.live_fire[:, pi], dt, co.live_crop[:, pi], co.live_graz[:, pi])
                self.nonforest[ri][:, j] = step["carbon"]
                for field in ("production", "fire", "crop_harvest", "grazing"):
                    ledger[field][:, bi] += step[field]
                ledger["live_to_cwd"][:, bi] += step["mortality"]*co.live_cwd_fraction[:, pi]
                ledger["live_to_soil"][:, bi] += step["mortality"]*(1-co.live_cwd_fraction[:, pi])
            # Reuse only the unchanged dead solver, with independent actual litter.
            dead = advance_dead_carbon(self.cwd[ri], self.soil[ri], self.hwp[ri],
                ledger["live_to_cwd"]/dt, ledger["live_to_soil"]/dt, np.zeros((2, 5, 2)), co.dead_rates, dt,
                member_ids=MEMBERS, biome_ids=BIOMES, product_ids=PRODUCTS)
            self.cwd[ri], self.soil[ri], self.hwp[ri] = dead.cwd.copy(), dead.soil.copy(), dead.hwp.copy()
            for field in ("cwd_to_soil", "cwd_respiration", "soil_respiration", "hwp_respiration"):
                ledger[field] = getattr(dead, field)
            ledger["atmospheric_loss"] = sum(ledger[f] for f in
                ("fire", "crop_harvest", "grazing", "cwd_respiration", "soil_respiration"))+dead.hwp_respiration.sum(axis=2)
            ledgers.append(ledger)
        self.time += dt
        return ledgers

    def events(self, local_plans, new_geometry, route):
        """Apply local transactions to copies; all operands use frozen stocks."""
        before = copy.deepcopy(self)
        ledgers = []
        for ri in range(2):
            m, e = 2, len(EVENTS)
            ledger = {"removed": np.zeros((m, 2, e)), "retained": np.zeros((m, 2, e)),
                "cwd_added": np.zeros((m, e)), "soil_added": np.zeros((m, e)),
                "hwp_added": np.zeros((m, e, 2)), "dead_transferred": np.zeros((m, 2, e))}
            biome_area = np.array([sum(before.area(ri, key) for key in self.geometry.keys if CLASS[key.current_class].biome == b) for b in BIOMES])
            stand_keys = set()
            for ci, plan in enumerate(local_plans):
                flow = plan.selected
                for ei, ki, raw_area in zip(flow.event_index, flow.source_index, flow.area):
                    key, spec = plan.opening.keys[ki], plan.events[ei]
                    area = raw_area*WEIGHTS[ci, ri]
                    if not area:
                        continue
                    sb, tb = BIOMES.index(CLASS[spec.source_class].biome), BIOMES.index(CLASS[spec.target_class].biome)
                    if CLASS[key.current_class].role == "nonforest":
                        ni = NF.index(key.current_class)
                        live = before.nonforest[ri][:, ni]*area/before.area(ri, key)
                        taken = np.column_stack((live*route.p_wood[:, sb], live*(1-route.p_wood[:, sb])))
                        self.nonforest[ri][:, ni] -= live
                    else:
                        local_area = before.geometry.area[ci, before.geometry.keys.index(key)]*WEIGHTS[ci, ri]
                        taken = before.local[ri][ci][key]*(area/local_area)
                        self.local[ri][ci][key] -= taken
                    ledger["removed"][:, :, ei] += taken
                    retained = taken if spec.vegetation_action == "retain" else np.zeros((2, 2))
                    ledger["retained"][:, :, ei] += retained
                    if spec.vegetation_action == "clear":
                        product = taken[:, 0, None]*route.p_hwp[:, sb]
                        residue = taken[:, 0]-product.sum(axis=1)
                        self.hwp[ri][:, sb] += product
                        self.cwd[ri][:, tb] += residue
                        self.soil[ri][:, tb] += taken[:, 1]
                        ledger["hwp_added"][:, ei] += product
                        ledger["cwd_added"][:, ei] += residue
                        ledger["soil_added"][:, ei] += taken[:, 1]
                    if CLASS[spec.target_class].role == "nonforest":
                        self.nonforest[ri][:, NF.index(spec.target_class)] += retained.sum(axis=1)
                    else:
                        stand = (spec.kind == "harvest" and spec.vegetation_action == "clear"
                                 and spec.source_class == spec.target_class == "secdf")
                        if stand:
                            dest = key
                            stand_keys.add(key)
                        else:
                            birth = key.birth_time if spec.age_action == "keep" else self.time
                            origin = key.entry_origin if spec.age_action == "keep" else ("cleared" if spec.vegetation_action == "clear" else key.current_class)
                            dest = SourceKey(spec.target_class, birth, origin)
                        self.local[ri][ci][dest] = self.local[ri][ci].get(dest, np.zeros((2, 2)))+retained
                    if spec.kind == "conversion":
                        for pool, field in enumerate(("cwd", "soil")):
                            transferred = getattr(before, field)[ri][:, sb]*area/biome_area[sb]
                            ledger["dead_transferred"][:, pool, ei] += transferred
                            if sb != tb:
                                getattr(self, field)[ri][:, sb] -= transferred
                                getattr(self, field)[ri][:, tb] += transferred
            # Explicit zero-support slots may remain until geometry removes them.
            for ci, cell in enumerate(self.local[ri]):
                for key in tuple(cell):
                    if key not in new_geometry.keys:
                        Checks().equal("vanished_local_stock", cell.pop(key), np.zeros((2, 2)),
                                       scale=abs(before.local[ri][ci].get(key, np.zeros((2, 2)))))
                for key in new_geometry.keys:
                    if CLASS[key.current_class].role != "nonforest":
                        cell.setdefault(key, np.zeros((2, 2)))
            # One regional cohort stores one stock. Unequal cell fractions would
            # otherwise leave densities the pool cannot represent on the next cut.
            if stand_keys:
                for key in stand_keys:
                    idx = new_geometry.keys.index(key)
                    pieces, areas = [], []
                    for ci in range(len(self.local[ri])):
                        pieces.append(np.asarray(self.local[ri][ci].get(key, np.zeros((2, 2))), float))
                        areas.append(float(new_geometry.area[ci, idx] * WEIGHTS[ci, ri]))
                    total_c = np.sum(pieces, axis=0)
                    total_a = sum(areas)
                    if total_a == 0:
                        continue
                    for ci, area in enumerate(areas):
                        self.local[ri][ci][key] = total_c * (area / total_a)
            ledgers.append(ledger)
        self.geometry = new_geometry
        return ledgers


def compare_states(checks, actual, oracle, ri):
    expected = oracle.aggregate(ri)
    a = {k: i for i, k in enumerate(actual.source_keys)}
    b = {k: i for i, k in enumerate(expected.source_keys)}
    for key in a.keys() | b.keys():
        checks.equal("source_area", actual.source_area[a[key]] if key in a else 0., expected.source_area[b[key]] if key in b else 0.)
    def forest_map(state):
        values = np.concatenate((state.forest.primary_carbon[:, :, None], state.forest.secondary_carbon), axis=2)
        return dict(zip(state.source_keys[:values.shape[2]], np.moveaxis(values, 2, 0)))
    ac, bc = forest_map(actual), forest_map(expected)
    for key in ac.keys() | bc.keys():
        checks.equal("forest_WF", ac.get(key, np.zeros((2, 2))), bc.get(key, np.zeros((2, 2))))
    for field in ("nonforest_carbon", "cwd", "soil", "hwp", "total_carbon"):
        checks.equal(field, getattr(actual, field), getattr(expected, field))
    assert actual.forest.member_ids == MEMBERS
    assert actual.forest.time == oracle.time
    assert actual.next_event_year == oracle.geometry.next_year


def run_case(strategy, age_on):
    geometry = initial_geometry(strategy)
    oracle = LocalOracle.start(geometry)
    actual = [oracle.aggregate(r) for r in range(2)]
    # Deliberately permute actual carbon slots independently of geometry.
    for ri, state in enumerate(actual):
        order = np.arange(len(state.cohort_class))[::-1]
        forest = replace(state.forest, birth_time=state.forest.birth_time[order],
            secondary_area=state.forest.secondary_area[order], secondary_carbon=state.forest.secondary_carbon[:, :, order],
            entry_origin=tuple(state.forest.entry_origin[i] for i in order))
        actual[ri] = replace(state, forest=forest, cohort_class=tuple(state.cohort_class[i] for i in order))
    co, rt = [coefficients(r) for r in range(2)], routing()
    checks = Checks()
    initial_total = np.stack([s.total_carbon for s in actual])
    accum = [{}, {}]
    expected_accum = [{}, {}]
    records = []
    event_total = np.zeros((2, 2, 2, len(EVENTS)))
    for year in range(2000, 2003):
        for half in range(2):
            expected_flux = oracle.continuous(co, age_on, .5)
            for ri in range(2):
                step = advance_land_carbon(actual[ri], co[ri], .5,
                    age_production=age_setting(ri) if age_on else None)
                actual[ri] = step.state
                for field, values in expected_flux[ri].items():
                    checks.equal("continuous_"+field, step.ledger[field], values)
                    accum[ri][field] = accum[ri].get(field, np.zeros_like(values))+step.ledger[field]
                    expected_accum[ri][field] = expected_accum[ri].get(field, np.zeros_like(values))+values
                compare_states(checks, actual[ri], oracle, ri)
            if half == 1:
                continue
            activity = np.array([[.12, .21, .13, .15, .2, .08, .11, .07, .1],
                                 [.08, .1, .24, .12, .16, .06, .09, .04, .08],
                                 [.1, .26, .08, .14, .18, .07, .1, .05, .09]])
            # Spatial event pattern changes each year without reading carbon.
            activity = np.roll(activity, year-2000, axis=0)
            new, regional = advance_geometry(geometry, EVENTS, activity, year=year,
                region_ids=REGIONS, region_fraction=WEIGHTS)
            local_plans = []
            for ci in range(3):
                one = GeometryState(CLASSES, (geometry.cell_ids[ci],), geometry.keys, geometry.area[ci:ci+1], year, strategy)
                _, plan = advance_geometry(one, EVENTS, activity[ci:ci+1], year=year,
                    region_ids=("local",), region_fraction=np.ones((1, 1)))
                local_plans.append(plan)
            expected_event = oracle.events(local_plans, new, rt)
            for ri in range(2):
                actual[ri], ledger = apply_events(actual[ri], regional, rt)
                for field, expected in expected_event[ri].items():
                    checks.equal("event_"+field, getattr(ledger, field), expected)
                compare_states(checks, actual[ri], oracle, ri)
                event_total[ri] += ledger.removed
                records.append({"year": year, "region": REGIONS[ri],
                    "events": list(ledger.event_ids), "selected_area": ledger.executed_area.tolist(),
                    **{k: v.tolist() for k, v in expected_event[ri].items()},
                    "actual_total_after_event": actual[ri].total_carbon.tolist(),
                    "oracle_total_after_event": oracle.aggregate(ri).total_carbon.tolist()})
            geometry = new
    for ri in range(2):
        for field, value in accum[ri].items():
            checks.equal("cumulative_"+field, value, expected_accum[ri][field])
        production = accum[ri]["production"].sum(axis=1)
        loss = accum[ri]["atmospheric_loss"].sum(axis=1)
        checks.equal("whole_chain_budget", actual[ri].total_carbon-initial_total[ri], production-loss,
            scale=np.maximum(initial_total[ri], actual[ri].total_carbon)+production+loss)
    return {"strategy": strategy, "age_response": age_on, "scalar_comparisons": checks.count,
        "maximum_error_in_float64_eps": max(checks.worst.values()), "error_by_quantity_in_eps": checks.worst,
        "events": records, "removed_WF_sum": event_total.tolist(),
        "cumulative_fluxes": [{k: v.tolist() for k, v in d.items()} for d in accum],
        "final_states": [s.snapshot() for s in actual]}


@pytest.mark.parametrize("strategy", ["proportional", "land_use_specific"])
@pytest.mark.parametrize("age_on", [False, True])
def test_cell_region_oracle(strategy, age_on):
    run_case(strategy, age_on)


def hand_example():
    """Frozen arithmetic witnesses, independent of constructor/router details."""
    checks = Checks()
    # Two equal-age origins: only the actual retained-origin cell is harvested.
    weights = np.array([.25, .75])
    retained_density = np.array([[12., 4.], [36., 12.]])
    selected = weights*.2
    taken = selected[:, None]*retained_density
    checks.equal("hand_selected_WF", taken, np.array([[.6, .2], [5.4, 1.8]]))
    product = taken[:, 0, None]*np.array([.2, .3])
    checks.equal("hand_HWP", product, np.array([[.12, .18], [1.08, 1.62]]))
    checks.equal("hand_CWD", taken[:, 0]-product.sum(axis=1), np.array([.3, 2.7]))
    checks.equal("hand_soil", taken[:, 1], np.array([.2, 1.8]))
    # Averaging across origins or regions must yield a visibly wrong answer.
    wrong_origin = selected[:, None]*(retained_density+np.array([[2., 1.], [6., 3.]]))/2
    wrong_region = selected[:, None]*np.sum(weights[:, None]*retained_density, axis=0)
    assert not np.allclose(taken, wrong_origin)
    assert not np.allclose(taken, wrong_region)
    step = analytic_live(np.array([2.]), np.array([.3]), np.array([.08]), np.array([.02]), .5)
    checks.equal("hand_live_end", step["carbon"], np.array([3-np.exp(-.05)]))
    from oscar._core._model._modules.physical_land import advance_live_carbon
    actual = advance_live_carbon(np.array([2.]), .3, .08, .02, .5)
    for name in ("carbon", "production", "mortality", "fire"):
        checks.equal("hand_live_"+name, getattr(actual, name), step[name])
    return {"actual_selected_WF": taken.tolist(), "HWP": product.tolist(),
            "birth_only_wrong_WF": wrong_origin.tolist(), "whole_cell_density_wrong_WF": wrong_region.tolist(),
            "constant_live_end": float(step["carbon"][0]), "scalar_comparisons": checks.count}


def test_independent_hand_examples():
    hand_example()


def exhaustion_case(kind):
    """Opening tolerance must not create an old key absent next year.

    The native variant changes no input areas by hand: regional carbon and
    cell-wise geometry obtain slightly different sums after ordinary events.
    """
    if kind == "literal_roundoff":
        geometry = GeometryState(CLASSES, (1,),
            (SourceKey("primf"), SourceKey("secdf", 1990., "cleared"), *(SourceKey(n) for n in NF)),
            [[0., .3, 1., 1., 1., 1., 1.]], 2000, "proportional")
        event = (EventSpec("full", "secdf", "secdf", "harvest", "clear", "reset", "proportional"),)
        new, plan = advance_geometry(geometry, event, [[.3]], year=2000,
            region_ids=(REGIONS[0],), region_fraction=[[1.]])
        forest = TwoPoolForestState(2000.5, MEMBERS, 0., np.zeros((2, 2)), [1990.],
            [.1+.2], np.ones((2, 2, 1)), REGIONS[0], ("cleared",))
        states = [LandEventState(forest, CLASSES, BIOMES, PRODUCTS, ("secdf",),
            np.ones(5), np.ones((2, 5)), np.zeros((2, 5)), np.zeros((2, 5)), np.zeros((2, 5, 2)), 2000)]
        weights, regions, year = np.ones((1, 1)), (REGIONS[0],), 2001
    elif kind == "native_three_cell":
        geometry = initial_geometry("proportional")
        local = LocalOracle.start(geometry)
        states = [replace(local.aggregate(r), forest=replace(local.aggregate(r).forest, time=2000.5)) for r in range(2)]
        activity = np.array([[.12, .21, .13, .15, .2, .08, .11, .07, .1],
                            [.08, .1, .24, .12, .16, .06, .09, .04, .08],
                            [.1, .26, .08, .14, .18, .07, .1, .05, .09]])
        geometry, first = advance_geometry(geometry, EVENTS, activity, year=2000,
            region_ids=REGIONS, region_fraction=WEIGHTS)
        for ri in range(2):
            states[ri], _ = apply_events(states[ri], first, routing())
            states[ri] = advance_land_carbon(states[ri], coefficients(ri), 1., age_production=None).state
        event = (EventSpec("clear_all", "secdf", "secdf", "harvest", "clear", "reset", "proportional"),)
        new, plan = advance_geometry(geometry, event, geometry.class_area[:, [1]], year=2001,
            region_ids=REGIONS, region_fraction=WEIGHTS)
        weights, regions, year = WEIGHTS, REGIONS, 2002
    else:
        raise ValueError(kind)
    ghosts = []
    for ri in range(len(states)):
        states[ri], _ = apply_events(states[ri], plan, routing())
        actual_stocks = np.concatenate((states[ri].forest.primary_carbon[:, :, None],
                                       states[ri].forest.secondary_carbon), axis=2)
        for ki, (key, area) in enumerate(zip(states[ri].source_keys, states[ri].source_area)):
            planned = plan.closing.area[ri, plan.closing.keys.index(key)] if key in plan.closing.keys else 0.
            if planned == 0. and area > 0.:
                ghosts.append({"region": regions[ri], "key": vars(key), "area": float(area),
                               "carbon_WF": actual_stocks[:, :, ki].tolist()})
    _, zero = advance_geometry(new, event, np.zeros((len(new.cell_ids), 1)), year=year,
        region_ids=regions, region_fraction=weights)
    failures = []
    for ri, state in enumerate(states):
        state = replace(state, forest=replace(state.forest, time=year+.5))
        try:
            apply_events(state, zero, routing())
        except ValueError as error:
            failures.append({"region": regions[ri], "message": str(error)})
    return {"case": kind, "passed": not failures and not ghosts,
            "unexpected_old_sources": ghosts, "next_zero_year_failures": failures}


@pytest.mark.parametrize("kind", ["literal_roundoff", "native_three_cell"])
def test_full_exhaustion_accepts_next_zero_year(kind):
    result = exhaustion_case(kind)
    assert result["passed"], result


def positive_remainder_case():
    """A genuine small remainder must keep both its label and actual W/F."""
    geometry = GeometryState(CLASSES, (1,),
        (SourceKey("primf"), SourceKey("secdf", 1990., "range"), *(SourceKey(n) for n in NF)),
        [[0., .3, 1., 1., 1., 1., 1.]], 2000, "proportional")
    remainder = 2.**-30
    amount = .3-remainder
    event = (EventSpec("near_full", "secdf", "secdf", "harvest", "clear", "reset", "proportional"),)
    new, plan = advance_geometry(geometry, event, [[amount]], year=2000,
        region_ids=(REGIONS[0],), region_fraction=[[1.]])
    carbon = np.array([[[1.], [2.]], [[3.], [4.]]])
    forest = TwoPoolForestState(2000.5, MEMBERS, 0., np.zeros((2, 2)), [1990.],
        [.1+.2], carbon, REGIONS[0], ("range",))
    state = LandEventState(forest, CLASSES, BIOMES, PRODUCTS, ("secdf",), np.ones(5),
        np.ones((2, 5)), np.zeros((2, 5)), np.zeros((2, 5)), np.zeros((2, 5, 2)), 2000)
    out, ledger = apply_events(state, plan, routing())
    checks = Checks()
    key = SourceKey("secdf", 1990., "range")
    si = out.source_keys.index(key)-1
    expected_stock = np.asarray(np.asarray(carbon[:, :, 0], np.longdouble)*
                                np.longdouble(remainder)/np.longdouble(.3), float)
    checks.equal("positive_remainder_area", out.forest.secondary_area[si], .3)
    checks.equal("positive_remainder_WF", out.forest.secondary_carbon[:, :, si], expected_stock)
    assert np.all(out.forest.secondary_carbon[:, :, si] > 0)
    checks.equal("near_full_removed", ledger.removed[:, :, 0], carbon[:, :, 0]-expected_stock)
    checks.equal("near_full_total", out.total_carbon, state.total_carbon)
    _, zero = advance_geometry(new, event, [[0.]], year=2001,
        region_ids=(REGIONS[0],), region_fraction=[[1.]])
    advanced = replace(out, forest=replace(out.forest, time=2001.5))
    final, _ = apply_events(advanced, zero, routing())
    checks.equal("next_zero_WF", final.forest.secondary_carbon, out.forest.secondary_carbon)
    return {"case": "positive_remainder", "passed": True, "remaining_area": remainder,
            "remaining_WF": expected_stock.tolist(), "next_zero_year_passed": True,
            "maximum_error_in_float64_eps": max(checks.worst.values())}


def parallel_exhaustion_case():
    """Two concurrent removals exhaust local sources, with regional sum tails."""
    from math import fsum
    geometry = GeometryState(CLASSES, (11, 22, 33),
        (SourceKey("primf"), SourceKey("secdf", 1990., "range"), *(SourceKey(n) for n in NF)),
        [[0., a, 1., 1., 1., 1., 1.] for a in (.1, .2, .3)], 2000, "proportional")
    events = tuple(EventSpec(name, "secdf", "secdf", "harvest", "clear", "reset", "oldest")
                   for name in ("first", "second"))
    areas = geometry.class_area[:, 1]
    new, plan = advance_geometry(geometry, events, np.column_stack((areas*.1, areas*.9)),
        year=2000, region_ids=REGIONS, region_fraction=WEIGHTS)
    local = LocalOracle.start(geometry)
    evidence, failures = [], []
    source = SourceKey("secdf", 1990., "range")
    _, zero = advance_geometry(new, events, np.zeros((3, 2)), year=2001,
        region_ids=REGIONS, region_fraction=WEIGHTS)
    for ri in range(2):
        state = local.aggregate(ri)
        state = replace(state, forest=replace(state.forest, time=2000.5))
        rows = (plan.selected.region_index == ri) & (plan.selected.source_index == plan.opening.keys.index(source))
        evidence.append({"region": REGIONS[ri], "opening": float(plan.opening.area[ri, plan.opening.keys.index(source)]),
                         "selected_sum": fsum(plan.selected.area[rows]), "closing_old_source": 0.})
        try:
            out, ledger = apply_events(state, plan, routing())
            assert source in out.source_keys
            assert out.source_area[out.source_keys.index(source)] == plan.opening.area[ri, plan.opening.keys.index(source)]
            Checks().equal("parallel_total", out.total_carbon, state.total_carbon)
            Checks().equal("parallel_removed", ledger.removed.sum(axis=2), state.forest.secondary_carbon[:, :, 0])
            apply_events(replace(out, forest=replace(out.forest, time=2001.5)), zero, routing())
        except (ValueError, AssertionError) as error:
            failures.append({"region": REGIONS[ri], "message": str(error)})
    return {"case": "parallel_exhaustion", "passed": not failures,
            "regional_sums": evidence, "failures": failures}


def test_real_positive_remainder_survives():
    positive_remainder_case()


def test_parallel_full_exhaustion_accepts_next_zero_year():
    result = parallel_exhaustion_case()
    assert result["passed"], result


def same_key_keep_case():
    """Closing includes the return flow and cannot stand for old survivors."""
    geometry = initial_geometry("proportional")
    local = LocalOracle.start(geometry)
    events = (EventSpec("same_key_keep", "secdf", "secdf", "conversion", "retain", "keep", "proportional"),)
    activity = np.array([[.2], [.15], [.25]])
    new, plan = advance_geometry(geometry, events, activity, year=2000,
        region_ids=REGIONS, region_fraction=WEIGHTS)
    _, zero = advance_geometry(new, events, np.zeros((3, 1)), year=2001,
        region_ids=REGIONS, region_fraction=WEIGHTS)
    checks = Checks()
    for ri in range(2):
        state = local.aggregate(ri)
        state = replace(state, forest=replace(state.forest, time=2000.5))
        expected_removal = np.zeros((2, 2))
        for ci in range(3):
            selected_fraction = activity[ci, 0]/geometry.class_area[ci, 1]
            for key, stock in local.local[ri][ci].items():
                if key.current_class == "secdf":
                    expected_removal += stock*selected_fraction
        out, ledger = apply_events(state, plan, routing())
        checks.equal("self_keep_removed", ledger.removed[:, :, 0], expected_removal)
        checks.equal("self_keep_retained", ledger.retained[:, :, 0], expected_removal)
        assert out.source_keys == state.source_keys
        checks.equal("self_keep_area", out.source_area, state.source_area)
        checks.equal("self_keep_WF", out.forest.secondary_carbon, state.forest.secondary_carbon)
        for field in ("nonforest_carbon", "cwd", "soil", "hwp", "total_carbon"):
            checks.equal("self_keep_"+field, getattr(out, field), getattr(state, field))
        apply_events(replace(out, forest=replace(out.forest, time=2001.5)), zero, routing())
    return {"case": "same_full_key_keep", "passed": True,
            "maximum_error_in_float64_eps": max(checks.worst.values())}


def nonforest_biome_exit_case():
    """All Non-Forest live/dead stock leaves; HWP stays on source biome."""
    keys = (SourceKey("primf"), SourceKey("secdf", 1990., "cleared"), *(SourceKey(n) for n in NF))
    geometry = GeometryState(CLASSES, (11, 22, 33), keys,
        [[1., 1., a, 2*a, 1., 1., 1.] for a in (.1, .2, .3)], 2000, "proportional")
    local = LocalOracle.start(geometry)
    events = tuple(EventSpec(n+"_exit", n, "secdf", "conversion", "clear", "reset", "proportional")
                   for n in ("range", "secdn"))
    new, plan = advance_geometry(geometry, events, geometry.class_area[:, [3, 4]], year=2000,
        region_ids=REGIONS, region_fraction=WEIGHTS)
    _, zero = advance_geometry(new, events, np.zeros((3, 2)), year=2001,
        region_ids=REGIONS, region_fraction=WEIGHTS)
    checks, rt, moved = Checks(), routing(), []
    for ri in range(2):
        state = local.aggregate(ri)
        state = replace(state, forest=replace(state.forest, time=2000.5))
        out, ledger = apply_events(state, plan, rt)
        carbon = state.nonforest_carbon[:, :2].sum(axis=1)
        wood = carbon*rt.p_wood[:, 1]
        fine = carbon*(1-rt.p_wood[:, 1])
        product = wood[:, None]*rt.p_hwp[:, 1]
        residue = wood-product.sum(axis=1)
        checks.equal("exit_biome_area", out.biome_area[1], 0.)
        checks.equal("exit_nonforest_live", out.nonforest_carbon[:, :2], np.zeros((2, 2)))
        checks.equal("exit_source_cwd", out.cwd[:, 1], np.zeros(2))
        checks.equal("exit_source_soil", out.soil[:, 1], np.zeros(2))
        checks.equal("exit_destination_cwd", out.cwd[:, 0], state.cwd[:, 0]+state.cwd[:, 1]+residue)
        checks.equal("exit_destination_soil", out.soil[:, 0], state.soil[:, 0]+state.soil[:, 1]+fine)
        checks.equal("exit_source_hwp", out.hwp[:, 1], product)
        checks.equal("exit_dead_transferred", ledger.dead_transferred.sum(axis=2),
                     np.column_stack((state.cwd[:, 1], state.soil[:, 1])))
        checks.equal("exit_total", out.total_carbon, state.total_carbon)
        apply_events(replace(out, forest=replace(out.forest, time=2001.5)), zero, rt)
        moved.append({"region": REGIONS[ri], "cwd": state.cwd[:, 1].tolist(),
                      "soil": state.soil[:, 1].tolist(), "hwp_source_biome": out.hwp[:, 1].tolist()})
    return {"case": "complete_nonforest_biome_exit", "passed": True, "transfers": moved,
            "maximum_error_in_float64_eps": max(checks.worst.values())}


def test_keep_returns_to_identical_full_key():
    same_key_keep_case()


def test_nonforest_biome_exit_moves_all_dead_stock():
    nonforest_biome_exit_case()

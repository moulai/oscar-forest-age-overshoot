"""Native regional land component with a jointly advanced W/F reference.

The reference contains densities on unit-area biome columns, never a second
actual land inventory. Native environmental responses and the fixed calibrated
production split are shared by reference and actual carbon. Only actual carbon
experiences land events and age-dependent production.
"""
from __future__ import annotations
from dataclasses import dataclass
import importlib
import weakref
import numpy as np
import xarray as xr

from ..._base.cls_main import Model
from ..._base.land_step import LandStep
from ..._base.fct_solve import f_max, safe_exp, safe_ratio
from .dynamic_reference import (BIOMES, LIVE_POOLS, POOL_BIOME, ReferenceAdapter,
                                ReferenceState, ReferenceCoefficients, advance_reference)
from .physical_land import physical_production, physical_loss_rates
from .forest_reference import reference_decay_rates
from .dead_carbon import DeadCarbonRates
from .regional_forest_age import STATE_NAMES, RegionalRates
from .regional_forest_age_native import (EXCHANGES, HWP_EXCHANGES, RegionalLandUpdate,
    RANK_EXCHANGES, rank_zero, validate_target_rule)
from .regional_forest_secondary_carbon import (SEC_EXCHANGES, SEC_UNITS, SEC_INPUTS, CANDIDATE,
    validate_secondary_contract, initial_secondary_exchange)
from .regional_forest_age_diagnostics import (AGE_EXCHANGES, initial_age_exchange,
    HARVEST_EXCHANGES, HARVEST_UNITS, initial_harvest_exchange)

REFERENCE_STATES = ("rf_ref_live", "rf_ref_cwd", "rf_ref_soil")
NATIVE_EXCHANGES = EXCHANGES + ("D_Aland", "D_Efire", "D_nbp", "D_Frss", "rf_ref_budget_residual")
ENVIRONMENT = ("D_CO2", "D_Tl", "r_Pl")


def prepare_native_forest_parameters(parameters, calibration_environment):
    """Prepare constants from primary parameters and a labeled calibration.

    This computes only required native environmental secondaries and the fixed
    W/F allocation/PI litter normalization. It does not reuse old one-pool PI
    carbon or bookkeeping inventories. Dynamic PI stocks are installed later.
    """
    from .OSCAR_landC_density import OSCAR_landC_density
    from .OSCAR_landC_bk import OSCAR_landC_bk
    p = parameters.copy(deep=False)
    required = ("b2_npp_CO2", "v_fire", "v_cwd", "g_cwd_T")
    # These are recalculated even if an old effective dataset supplied them.
    p = p.drop_vars([name for name in required if name in p])
    p = xr.merge([p, OSCAR_landC_density.copy(only=required)._get_Par2(p)])
    p = p.drop_vars("p_hwp", errors="ignore")
    p = xr.merge([p, OSCAR_landC_bk.copy(only=["p_hwp"])._get_Par2(p)])
    adapter = ReferenceAdapter(p, calibration_environment, allocation_mode="fixed_calibration")
    p["rf_beta"] = adapter.beta
    p["rf_alpha_cal"] = adapter.alpha_cal
    p["rf_fall_pi"] = adapter.anchor
    p["rf_p_wood"] = xr.broadcast(p.p_wood, p.v_mort)[0].transpose("config", "reg_land", "bio_land")
    p["rf_p_hwp"] = xr.broadcast(p.p_hwp, p.rf_p_wood)[0].transpose("config", "reg_land", "bio_land", "box_hwp")
    zero = xr.zeros_like(p.rf_p_wood)
    # Cold-start dynamic spin-up uses a theoretical zero net-uptake baseline.
    p["rf_NBP_pi"] = zero.sum("bio_land")
    p["rf_ref_nbp_pi"] = zero
    p["Efire_pi"] = zero
    p.attrs["regional_forest_baseline"] = "cold start; install dynamic preindustrial baseline before historical coupling"
    return p.drop_vars("year", errors="ignore")


def _numeric(value, parameters):
    return xr.broadcast(value, parameters.rf_p_wood)[0].transpose("config", "reg_land", "bio_land").values


def prepare_native_coefficients(variables, parameters, layout):
    """Use native Var/Par environment, with no hidden parameter snapshot.

    This is the fixed-calibration ReferenceAdapter algebra with its constants
    already in Par. Pure physical_production/loss and reference_decay_rates
    supply all environmental responses, including their documented boundaries.
    """
    p = parameters
    if tuple(p.config.values) != layout.members or tuple(p.source_config.values) != layout.parameter_ids:
        raise ValueError("native and original parameter member identities differ from layout")
    if tuple(p.reg_land.values) != layout.regions or tuple(p.bio_land.values) != BIOMES:
        raise ValueError("native parameter region/biome identities differ from layout")
    environment = variables[list(ENVIRONMENT)]
    production = _numeric(physical_production(environment, p).production_input, p)
    loss = physical_loss_rates(environment, p)
    mu, fire = _numeric(loss.mortality_rate, p), _numeric(loss.fire_rate, p)
    beta, wood = _numeric(p.rf_beta, p), _numeric(p.p_wood, p)
    alpha = _numeric(p.rf_alpha_cal, p)[..., 0]
    prod = production[..., POOL_BIOME].copy()
    prod[..., 0] *= alpha
    prod[..., 1] *= 1-alpha
    mortality = mu[..., POOL_BIOME].copy()
    mortality[..., 0] *= beta[..., 0] / wood[..., 0]
    mortality[..., 1] *= (1-beta[..., 0]) / (1-wood[..., 0])
    route = beta[..., POOL_BIOME].copy()
    route[..., 0], route[..., 1] = 1., 0.
    decay = reference_decay_rates(environment, p, p.rf_fall_pi, p.rf_fall_pi)
    return ReferenceCoefficients(layout.parameter_ids, layout.regions, "fixed_calibration",
        prod, mortality, fire[..., POOL_BIOME],
        _numeric(p.p_charv, p)[..., POOL_BIOME], _numeric(p.p_graz, p)[..., POOL_BIOME], route,
        _numeric(decay.cwd_rate, p), _numeric(decay.soil_rate, p),
        _numeric(p.x_resp_fall, p), _numeric(f_max(xr.where(p.v_resp > 0, p.v_resp, 1.)), p),
        _numeric(p.rf_fall_pi, p), _numeric(p.p_cwd_resp, p))


def _as_crb(value):
    """Move a parameter/environment array onto (config, reg_land, bio_land) axes."""
    order = ("config", "reg_land", "bio_land")
    if not hasattr(value, "dims"):
        return np.asarray(value, dtype=float).reshape((1, 1, 1))
    present = [dim for dim in order if dim in value.dims]
    if any(dim not in order for dim in value.dims):
        raise ValueError("unexpected dimensions: " + str(value.dims))
    data = np.asarray(value.transpose(*present).values, dtype=float) if present else np.asarray(value.values, dtype=float)
    shape = tuple(value.sizes[dim] if dim in value.dims else 1 for dim in order)
    return np.broadcast_to(data.reshape(shape), shape)


class _ObjectCache:
    """Cache by object identity. Datasets are unhashable, so WeakKeyDictionary cannot."""

    def __init__(self):
        self._data = {}

    def _drop(self, _ref):
        dead = [key for key, (ref, _) in self._data.items() if ref() is None]
        for key in dead:
            del self._data[key]

    def get(self, obj):
        item = self._data.get(id(obj))
        if item is None or item[0]() is not obj:
            return None
        return item[1]

    def put(self, obj, value):
        self._data[id(obj)] = (weakref.ref(obj, self._drop), value)


_RATE_CACHE = _ObjectCache()
_HWP_CACHE = _ObjectCache()


def _rate_arrays(parameters):
    """Parameter factors in the same order _numeric() broadcasts to rf_p_wood."""
    cached = _RATE_CACHE.get(parameters)
    if cached is not None:
        return cached
    p = parameters
    arrays = {}
    for name in ("k_npp", "npp_piL", "b_npp_CO2", "x_npp_CO2", "g_npp_T", "g_npp_T2", "x_npp_P",
                 "b2_npp_CO2", "v_mort", "v_fire", "x_mort_npp", "x_fire_npp", "g_mort_T", "x_mort_P",
                 "x_fire_CO2", "g_fire_T", "g_fire_P", "Pl_pi", "v_cwd", "g_cwd_T", "v_resp",
                 "x_resp_fall", "g_resp_T", "x_resp_P", "p_charv", "p_graz", "p_wood", "rf_beta",
                 "rf_fall_pi", "p_cwd_resp"):
        arrays[name] = np.array(_numeric(p[name], p), dtype=float, copy=True)
    arrays["CO2_pi"] = np.array(_numeric(p.CO2_pi, p), dtype=float, copy=True)
    arrays["CO2_piL"] = np.array(_numeric(p.CO2_piL, p), dtype=float, copy=True)
    alpha = np.array(_numeric(p.rf_alpha_cal, p)[..., 0], dtype=float, copy=True)
    arrays["alpha"] = alpha
    _RATE_CACHE.put(parameters, arrays)
    return arrays


def _environment_block(variables, arrays):
    """Same production, loss and reference-decay algebra as the xarray helpers."""
    co2 = _as_crb(variables.D_CO2)
    temperature_anomaly = _as_crb(variables.D_Tl)
    precipitation = _as_crb(variables.r_Pl)
    co2_pi = arrays["CO2_pi"]
    co2_pil = arrays["CO2_piL"]
    absolute = co2_pi + co2
    log_ratio = np.log(absolute / co2_pil)
    x = arrays["x_npp_CO2"]
    denominator = np.where(x == 0, 1.0, x)
    response = np.where(x == 0, log_ratio, np.expm1(x * log_ratio) / denominator)
    co2_factor = 1.0 + arrays["b_npp_CO2"] * response
    temperature = safe_exp(arrays["g_npp_T"] * temperature_anomaly, 100) * np.exp(
        -arrays["g_npp_T2"] * np.maximum(temperature_anomaly, 0) ** 2)
    rain = safe_exp(arrays["x_npp_P"] * np.log(precipitation), 100)
    raw = arrays["k_npp"] * arrays["npp_piL"] * co2_factor * temperature * rain
    production = np.where(raw >= 0, raw, 0.0)

    invalid = (1.0 + arrays["b_npp_CO2"] * np.where(
        x == 0, np.log(co2_pi / co2_pil), np.expm1(x * np.log(co2_pi / co2_pil)) / denominator)) <= 0
    if np.any(invalid & ~((arrays["x_mort_npp"] == 0) & (arrays["x_fire_npp"] == 0))):
        raise ValueError("Nonpositive PI production normalization with nonzero NPP loss feedback")
    b2 = np.where(invalid, 0.0, arrays["b2_npp_CO2"])
    x_safe = np.where(invalid, 1.0, x)
    mort_safe = np.where(arrays["v_mort"] > 0, arrays["v_mort"], 1.0)
    fire_safe = np.where(arrays["v_fire"] > 0, arrays["v_fire"], 1.0)
    fct_co2 = 1 + b2 / x_safe * ((1 + co2 / co2_pi) ** x_safe - 1)
    fct_tl = safe_exp(arrays["g_npp_T"] * temperature_anomaly, 100) * np.exp(
        -arrays["g_npp_T2"] * np.maximum(temperature_anomaly, 0) ** 2)
    fct_pl = safe_exp(arrays["x_npp_P"] * np.log(precipitation), 100)
    ratio = np.where(invalid, 1.0, safe_ratio(fct_co2 * fct_tl * fct_pl))
    mort_cap = f_max(mort_safe)
    fire_cap = f_max(fire_safe)
    r_mort = (safe_exp(arrays["x_mort_npp"] * np.log(ratio), mort_cap)
              * safe_exp(arrays["g_mort_T"] * temperature_anomaly, mort_cap)
              * safe_exp(arrays["x_mort_P"] * np.log(precipitation), mort_cap))
    r_fire = (safe_exp(arrays["x_fire_npp"] * np.log(ratio), fire_cap)
              * safe_exp(arrays["x_fire_CO2"] * np.log1p(co2 / co2_pi), fire_cap)
              * safe_exp(arrays["g_fire_T"] * temperature_anomaly, fire_cap)
              * safe_exp(arrays["g_fire_P"] * arrays["Pl_pi"] * (precipitation - 1), fire_cap))
    mortality = arrays["v_mort"] * r_mort
    fire = arrays["v_fire"] * r_fire
    cwd_safe = np.where(arrays["v_cwd"] > 0, arrays["v_cwd"], 1.0)
    resp_safe = np.where(arrays["v_resp"] > 0, arrays["v_resp"], 1.0)
    cwd_rate = arrays["v_cwd"] * safe_exp(arrays["g_cwd_T"] * temperature_anomaly, f_max(cwd_safe))
    cap = f_max(resp_safe)
    # Coefficient preparation evaluates the litter response at the anchor itself.
    anchor_ratio = np.ones(arrays["rf_fall_pi"].shape)
    input_response = safe_exp(arrays["x_resp_fall"] * np.log(safe_ratio(anchor_ratio)), cap)
    soil_rate = arrays["v_resp"] * input_response * safe_exp(arrays["g_resp_T"] * temperature_anomaly, cap) * safe_exp(
        arrays["x_resp_P"] * np.log(precipitation), cap)
    return production, mortality, fire, cwd_rate, soil_rate


def fast_native_coefficients(variables, parameters, layout):
    """prepare_native_coefficients without rebuilding xarray expressions each step."""
    if tuple(parameters.config.values) != layout.members or tuple(parameters.source_config.values) != layout.parameter_ids:
        raise ValueError("native and original parameter member identities differ from layout")
    if tuple(parameters.reg_land.values) != layout.regions or tuple(parameters.bio_land.values) != BIOMES:
        raise ValueError("native parameter region/biome identities differ from layout")
    arrays = _rate_arrays(parameters)
    production, mortality, fire, cwd_rate, soil_rate = _environment_block(variables, arrays)
    alpha = arrays["alpha"]
    prod = production[..., POOL_BIOME].copy()
    prod[..., 0] *= alpha
    prod[..., 1] *= 1 - alpha
    mu = mortality[..., POOL_BIOME].copy()
    beta = arrays["rf_beta"]
    wood = arrays["p_wood"]
    mu[..., 0] *= beta[..., 0] / wood[..., 0]
    mu[..., 1] *= (1 - beta[..., 0]) / (1 - wood[..., 0])
    route = beta[..., POOL_BIOME].copy()
    route[..., 0], route[..., 1] = 1., 0.
    resp_safe = np.where(arrays["v_resp"] > 0, arrays["v_resp"], 1.0)
    return ReferenceCoefficients(layout.parameter_ids, layout.regions, "fixed_calibration",
        prod, mu, fire[..., POOL_BIOME],
        arrays["p_charv"][..., POOL_BIOME], arrays["p_graz"][..., POOL_BIOME], route,
        cwd_rate, soil_rate, arrays["x_resp_fall"], f_max(resp_safe),
        arrays["rf_fall_pi"], arrays["p_cwd_resp"])


def cached_hwp_loss(parameters, template):
    """Product decay does not depend on the running state, only on its shape."""
    cached = _HWP_CACHE.get(parameters)
    if cached is None:
        cached = xr.broadcast(parameters.v_hwp, template.rf_C_hwp)[0].transpose(
            "config", "reg_land", "bio_land", "box_hwp").values.copy()
        _HWP_CACHE.put(parameters, cached)
    return cached


def biome_area(state, layout):
    """Actual area by native biome, preserving source classes in the state."""
    value = np.zeros((len(layout.members), len(layout.regions), len(layout.biomes)))
    value[:, :, 0] = state.rf_A_primary.values + state.rf_A_forest.sum(("forest_class", "birth_time")).values
    mapping = {c.class_id: layout.biomes.index(c.biome) for c in layout.classes}
    for j, name in enumerate(layout.nonforest_classes):
        value[:, :, mapping[name]] += state.rf_A_nonforest.values[:, :, j]
    return xr.DataArray(value, dims=("config", "reg_land", "bio_land"),
                        coords=dict(config=list(layout.members), reg_land=list(layout.regions), bio_land=list(layout.biomes)))


def carbon_by_biome(state, layout):
    value = np.zeros(state.rf_C_cwd.shape, dtype=float)
    forest = layout.biomes[0]
    forest_index = list(state.bio_land.values).index(forest)
    value[:, :, forest_index] = (state.rf_W_primary.values + state.rf_F_primary.values
        + (state.rf_W_forest.values + state.rf_F_forest.values).sum(axis=(2, 3)))
    mapping = {c.class_id: c.biome for c in layout.classes}
    biome_index = {name: i for i, name in enumerate(state.bio_land.values)}
    class_index = {name: i for i, name in enumerate(layout.nonforest_classes)}
    nonforest = state.rf_C_nonforest.values
    for name in layout.nonforest_classes:
        value[:, :, biome_index[mapping[name]]] += nonforest[:, :, class_index[name]]
    return xr.DataArray(value, dims=state.rf_C_cwd.dims, coords=state.rf_C_cwd.coords)


def initialize_native_forest_state(layout, initial_area, *, year, reference=None):
    """Initialize primary/nonforest cover without inventing secondary ages.

    reference=None starts actual and reference carbon at zero for dynamic PI
    spin-up. A supplied density Dataset explicitly initializes actual stocks
    by density times area; this is only an initialization operation.
    """
    if layout.biomes != BIOMES:
        raise ValueError("native reference requires the five OSCAR biomes")
    if tuple(initial_area.reg_land.values) != layout.regions or tuple(initial_area.luh_class.values) != tuple(c.class_id for c in layout.classes):
        raise ValueError("initial area coordinates differ from the prepared source classes")
    if np.any(initial_area < 0) or not np.isfinite(initial_area).all():
        raise ValueError("initial area must be finite and nonnegative")
    if np.any(initial_area.sel(luh_class=list(layout.forest_classes)) != 0):
        raise ValueError("nonzero initial cohort area requires an explicit existing age state")
    if "config" not in initial_area.dims:
        initial_area = initial_area.expand_dims(config=list(layout.members)).copy(deep=True)
    if tuple(initial_area.config.values) != layout.members:
        raise ValueError("Initial area member identities differ from layout")
    initial_area = initial_area.transpose("config", "reg_land", "luh_class")
    state = layout.empty_state(year).assign_coords(ref_pool=list(LIVE_POOLS))
    state["rf_ref_live"] = (("config", "reg_land", "ref_pool"), np.zeros((len(layout.members), len(layout.regions), 6)))
    for name in REFERENCE_STATES[1:]:
        state[name] = xr.zeros_like(state.rf_C_cwd)
    if reference is not None:
        for name in REFERENCE_STATES:
            supplied, _ = xr.align(reference[name], state[name], join="exact")
            state[name] = supplied.transpose(*state[name].dims).drop_vars("year", errors="ignore")
    for name in REFERENCE_STATES:
        state[name].attrs["units"] = "PgC Mha-1"
    state["rf_A_primary"] = initial_area.sel(luh_class=layout.primary_class, drop=True)
    state["rf_A_nonforest"] = initial_area.sel(luh_class=list(layout.nonforest_classes)).rename(luh_class="nonforest_class")
    state["rf_W_primary"] = state.rf_ref_live.isel(ref_pool=0, drop=True) * state.rf_A_primary
    state["rf_F_primary"] = state.rf_ref_live.isel(ref_pool=1, drop=True) * state.rf_A_primary
    mapping = {c.class_id: layout.biomes.index(c.biome) for c in layout.classes}
    for name in layout.nonforest_classes:
        state.rf_C_nonforest.loc[dict(nonforest_class=name)] = state.rf_ref_live.isel(ref_pool=mapping[name]+1, drop=True) * state.rf_A_nonforest.sel(nonforest_class=name, drop=True)
    area = biome_area(state, layout)
    state["rf_C_cwd"], state["rf_C_soil"] = state.rf_ref_cwd * area, state.rf_ref_soil * area
    for name in STATE_NAMES:
        state[name].attrs["units"] = "Mha" if name.startswith("rf_A") else "PgC"
    layout.validate(state, year)
    return state


def _reference_state(state, layout, time):
    return ReferenceState(layout.parameter_ids, layout.regions, time,
                          state.rf_ref_live.transpose("config", "reg_land", "ref_pool").values,
                          state.rf_ref_cwd.transpose("config", "reg_land", "bio_land").values,
                          state.rf_ref_soil.transpose("config", "reg_land", "bio_land").values)


def install_preindustrial_baseline(parameters, state, layout):
    """Install the same dynamic W/F PI state for land, fire and area feedback.

    The mathematical no-activity mature net uptake is zero. Any finite spin-up
    drift is reported explicitly, never fitted away as a trajectory offset.
    """
    p = parameters.copy(deep=False)
    pi = xr.Dataset(dict(D_CO2=0., D_Tl=0., r_Pl=1.))
    c = prepare_native_coefficients(pi, p, layout)
    ref = _reference_state(state, layout, float(state.year))
    instantaneous = advance_reference(ref, c, 0.)
    area = biome_area(state, layout)
    p["Aland_pi"] = area
    def density(value):
        return xr.DataArray(value, dims=("config", "reg_land", "bio_land"),
                            coords={key: state[key] for key in ("config", "reg_land", "bio_land")})
    p["cveg_pi"], p["ccwd_pi"], p["csoil_pi"] = density(ref.live_by_biome), density(ref.cwd), density(ref.soil)
    p["npp_pi"] = physical_production(pi, p).production_input
    fire_density = c.live_fire * ref.live
    p["Efire_pi"] = density(np.concatenate((fire_density[..., :2].sum(axis=-1, keepdims=True),
                                             fire_density[..., 2:]), axis=-1)) * area
    p["rf_NBP_pi"] = xr.zeros_like(state.rf_W_primary)
    p["rf_ref_nbp_pi"] = xr.zeros_like(state.rf_C_cwd)
    effective = c.live_production * (1-c.live_crop-c.live_graz)
    live_loss = c.live_fire * ref.live
    live_net = effective-live_loss
    nbp_density = np.concatenate((live_net[..., :2].sum(axis=-1, keepdims=True), live_net[..., 2:]), axis=-1)
    nbp_density -= c.cwd_resp_fraction * c.cwd_loss * ref.cwd + instantaneous.soil_rate * ref.soil
    p["rf_pi_NBP_drift"] = (density(nbp_density) * area).sum("bio_land")
    p.attrs["regional_forest_baseline"] = "dynamic mature PI W/F reference; theoretical NBP zero; finite drift separately reported"
    return p.drop_vars("year", errors="ignore")


@dataclass(frozen=True)
class NativeForestUpdate:
    layout: object
    age_production: tuple | None = None
    preference: object = None
    hwp_target_events: tuple | None = None
    purpose_rank: dict | None = None
    hwp_target_rule: str = "weighted"
    secondary_agb_events: tuple | None = None
    harvest_projection: str | None = None
    secondary_harvest_contract: str = CANDIDATE
    age_clock: object = None

    def __call__(self, variables, parameters, forcing, t0, t1, event):
        layout, p, dt = self.layout, parameters, t1-t0
        c = fast_native_coefficients(variables, p, layout)
        reference = advance_reference(_reference_state(variables, layout, t0), c, dt)
        template = variables.rf_C_cwd.drop_vars("year", errors="ignore")
        ref_nbp = xr.DataArray((reference.ledger["production"] - reference.ledger["atmospheric_loss"])/dt,
                              dims=template.dims, coords=template.coords) - p.rf_ref_nbp_pi
        product_loss = cached_hwp_loss(p, variables)
        dead = tuple(DeadCarbonRates(layout.members, layout.biomes, layout.products,
                                     reference.cwd_rate[:, ri], reference.soil_rate[:, ri],
                                     c.cwd_resp_fraction[:, ri], product_loss[:, ri])
                     for ri in range(len(layout.regions)))
        rates = RegionalRates(c.live_production, c.live_mortality, c.live_fire,
                              c.live_crop, c.live_graz, c.live_cwd_fraction, dead)
        actual_update = RegionalLandUpdate(layout, lambda *args: rates,
            lambda state, *args: (ref_nbp * biome_area(state, layout)).sum("bio_land"),
            self.age_production, self.preference, self.hwp_target_events, self.purpose_rank, self.hwp_target_rule, self.secondary_agb_events, self.harvest_projection, self.secondary_harvest_contract, self.age_clock)
        state, exchanges = actual_update(variables, p, forcing, t0, t1, event)
        state.update({name: (variables[name].dims, values) for name, values in (
            ("rf_ref_live", reference.state.live), ("rf_ref_cwd", reference.state.cwd),
            ("rf_ref_soil", reference.state.soil))})
        state = state.assign_coords(ref_pool=list(LIVE_POOLS))
        area_change = biome_area(state, layout) - p.Aland_pi
        exchanges.update({
            "D_Aland": area_change,
            "D_Efire": exchanges.rf_Efire - p.Efire_pi,
            "D_nbp": ref_nbp,
            "D_Frss": (ref_nbp * area_change).sum("bio_land"),
            "rf_ref_budget_residual": xr.DataArray(reference.ledger["budget_residual"]/dt,
                                                    dims=template.dims, coords=template.coords),
        })
        return state, exchanges


def build_native_forest_age_model(layout, *, age_production=None, preference=None, hwp_target_events=None, purpose_rank=None, hwp_target_rule="weighted", secondary_agb_events=None, harvest_projection=None, secondary_harvest_contract=CANDIDATE, age_clock=None):
    """Build the land component; Ini/Par/For remain ordinary native datasets."""
    validate_target_rule(layout, purpose_rank, hwp_target_events, hwp_target_rule)
    validate_secondary_contract(layout, secondary_agb_events, harvest_projection,
                                hwp_target_events, purpose_rank, hwp_target_rule, secondary_harvest_contract)
    if layout.biomes != BIOMES or layout.source_members is None:
        raise ValueError("native reference requires five native biomes and explicit original member IDs")
    model = Model("OSCAR_landC_forest_age")
    states = STATE_NAMES + REFERENCE_STATES
    dependencies = states + ("event_area",) + ENVIRONMENT
    if hwp_target_events is not None:
        dependencies += ("HWP_target",)
    if secondary_agb_events is not None:
        dependencies += SEC_INPUTS
    exchanges = (RANK_EXCHANGES if purpose_rank is not None else ()) + NATIVE_EXCHANGES + (HWP_EXCHANGES + ("rf_HWP_managed_area",) if hwp_target_events is not None else ())
    exchanges += SEC_EXCHANGES if secondary_agb_events is not None else ()
    template = layout.empty_state(0.)
    for name in states:
        dims = (("config", "reg_land", "ref_pool") if name == "rf_ref_live" else
                ("config", "reg_land", "bio_land") if name in REFERENCE_STATES else template[name].dims)
        model.process(name, (name, "event_area") + ENVIRONMENT, Eq=lambda V, P, n=name: V[n],
                      units="PgC Mha-1" if name in REFERENCE_STATES else template[name].attrs["units"],
                      core_dims=[d for d in dims if d != "config"])
    def initial_exchange(name, V, P):
        if name in SEC_EXCHANGES:
            return initial_secondary_exchange(name, V, secondary_agb_events)
        if name in RANK_EXCHANGES:
            return rank_zero(layout, hwp_target_events or tuple(purpose_rank))
        if name in HARVEST_EXCHANGES:
            return initial_harvest_exchange(name, layout)
        if name == "rf_HWP_managed_area":
            return xr.zeros_like(V.rf_W_primary).expand_dims(managed_event=list(hwp_target_events), axis=-1)
        if name in HWP_EXCHANGES:
            return xr.zeros_like(V.rf_W_primary)
        if name in AGE_EXCHANGES:
            return initial_age_exchange(name, V, layout, float(V.year))
        if name == "D_Aland":
            return biome_area(V, layout) - P.Aland_pi
        if name in ("rf_forest_W_mean", "rf_forest_F_mean"):
            pool = "W" if name == "rf_forest_W_mean" else "F"
            return V["rf_"+pool+"_primary"] + V["rf_"+pool+"_forest"].sum(("forest_class", "birth_time"))
        if name == "rf_forest_area_mean":
            return biome_area(V, layout).sel(bio_land="Forest", drop=True)
        if name in ("D_Fland", "D_Eluc", "D_Frss", "rf_NBP", "rf_HWP_input", "rf_NPP_W", "rf_NPP_F"):
            return xr.zeros_like(V.rf_W_primary)
        if name == "rf_event_count":
            return xr.DataArray(0.)
        if name == "rf_budget_residual":
            return xr.zeros_like(V.rf_W_primary.sum("reg_land"))
        return xr.zeros_like(V.rf_C_cwd)
    for name in exchanges:
        units = (SEC_UNITS[name] if name in SEC_EXCHANGES else "yr yr-1" if name in RANK_EXCHANGES else HARVEST_UNITS[name] if name in HARVEST_EXCHANGES else "Mha" if name in ("D_Aland", "rf_forest_area_mean", "rf_age_area_mean") else
                 "Mha yr-1" if name in ("rf_event_area_by_age", "rf_HWP_managed_area") else "PgC" if name.endswith("_mean") else "yr-1" if name == "rf_event_count" else
                 "PgC Mha-1 yr-1" if name in ("D_nbp", "rf_ref_budget_residual") else "PgC yr-1")
        model.process(name, dependencies, Eq=lambda V, P, n=name: initial_exchange(n, V, P), units=units)
    model.process("Aland", ("D_Aland",), Eq=lambda V, P: P.Aland_pi + V.D_Aland, units="Mha")
    model.process("Cveg", STATE_NAMES, Eq=lambda V, P: carbon_by_biome(V, layout), units="PgC")
    for out, source in (("Ccwd", "rf_C_cwd"), ("Csoil", "rf_C_soil")):
        model.process(out, (source,), Eq=lambda V, P, n=source: V[n], units="PgC")
    model.process("Chwp", ("rf_C_hwp",), Eq=lambda V, P: V.rf_C_hwp.sum("box_hwp"), units="PgC")
    model.process("Cland", ("Cveg", "Ccwd", "Csoil", "Chwp"), Eq=lambda V, P: V.Cveg+V.Ccwd+V.Csoil+V.Chwp, units="PgC")
    for out, source in (("NBP", "rf_NBP_biome"), ("NPP", "rf_NPP"), ("Efire", "rf_Efire"), ("Eresp", "rf_Eresp"), ("Ehwp", "rf_Ehwp")):
        model.process(out, (source,), Eq=lambda V, P, n=source: V[n], units="PgC yr-1")
    model.register_land_step(LandStep(states, exchanges, NativeForestUpdate(layout, age_production, preference, hwp_target_events, purpose_rank, hwp_target_rule, secondary_agb_events, harvest_projection, secondary_harvest_contract, age_clock)))
    return model


def assemble_native_forest_age_model(layout, *, age_production=None, preference=None, modules=None, hwp_target_events=None, purpose_rank=None, hwp_target_rule="weighted", secondary_agb_events=None, harvest_projection=None, secondary_harvest_contract=CANDIDATE, age_clock=None):
    """Replace native land/bk and its fire exchange, keeping all other modules.

    modules optionally restricts components for controlled scientific tests.
    Full assembly uses the standard OSCAR module list, without an external loop.
    """
    if modules is None:
        from ..OSCAR import list_modules
        modules = tuple(name for name in list_modules if name != "landC")
    if "landC" in modules or len(set(modules)) != len(modules):
        raise ValueError("do not merge the old land bookkeeping component or duplicate modules")
    model = build_native_forest_age_model(layout, age_production=age_production, preference=preference,
                                         hwp_target_events=hwp_target_events, purpose_rank=purpose_rank, hwp_target_rule=hwp_target_rule,
                                         secondary_agb_events=secondary_agb_events, harvest_projection=harvest_projection,
                                         secondary_harvest_contract=secondary_harvest_contract, age_clock=age_clock)
    for name in modules:
        imported = importlib.import_module("."+"OSCAR_"+name, __package__)
        component = getattr(imported, "OSCAR_"+name)
        if name == "fire":
            component = component.copy(only=[key for key in component.proc_all if key != "D_Efire"])
        model = model.merge(component, new_name="OSCAR_forest_age")
    return model

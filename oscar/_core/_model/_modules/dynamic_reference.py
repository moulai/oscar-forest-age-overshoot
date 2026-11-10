"""Environment-only reference densities with one conservative physical ledger.

The reference has six live pools (forest wood/fine plus four other biomes),
CWD and soil per native biome. It has no land events, age state or HWP stock.
Units are Pg C Mha-1, years, and Pg C Mha-1 yr-1. Each region/biome is a
separate unit-area column; summing these densities is not a global carbon sum.
"""
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..._base.fct_solve import safe_exp, safe_ratio, f_max
from .physical_land import physical_production, physical_loss_rates, mortality_partition, advance_live_carbon
from .forest_reference import preindustrial_fall_anchor, reference_decay_rates
from .dead_carbon import DeadCarbonRates, advance_dead_carbon

BIOMES = ("Forest", "Non-Forest", "Cropland", "Pasture", "Urban")
LIVE_POOLS = ("Forest_wood", "Forest_fine", *BIOMES[1:])
POOL_BIOME = np.array([0, 0, 1, 2, 3, 4])


def _ids(values, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(name + " must be nonempty and unique")
    return values


def _array(value, shape, name, nonnegative=True):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all() or (nonnegative and np.any(result < 0)):
        raise ValueError(name + " has invalid shape, values or sign")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReferenceState:
    """Physical densities only; no raw parameters or analytic mature stocks."""
    member_ids: tuple
    region_ids: tuple
    time: float
    live: np.ndarray
    cwd: np.ndarray
    soil: np.ndarray

    def __post_init__(self):
        for name in ("member_ids", "region_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if not np.isfinite(self.time):
            raise ValueError("time must be finite")
        shape = (len(self.member_ids), len(self.region_ids))
        for name, n in (("live", 6), ("cwd", 5), ("soil", 5)):
            object.__setattr__(self, name, _array(getattr(self, name), shape + (n,), name))

    @property
    def live_by_biome(self):
        return np.concatenate((self.live[..., :2].sum(axis=-1, keepdims=True), self.live[..., 2:]), axis=-1)

    @property
    def carbon_by_biome(self):
        return self.live_by_biome + self.cwd + self.soil


@dataclass(frozen=True)
class ReferenceCoefficients:
    """Prepared midpoint environment; all arrays use explicit member/region axes."""
    member_ids: tuple
    region_ids: tuple
    allocation_mode: str
    live_production: np.ndarray
    live_mortality: np.ndarray
    live_fire: np.ndarray
    live_crop: np.ndarray
    live_graz: np.ndarray
    live_cwd_fraction: np.ndarray
    cwd_loss: np.ndarray
    soil_environment_loss: np.ndarray
    soil_input_exponent: np.ndarray
    soil_input_cap: np.ndarray
    fall_anchor: np.ndarray
    cwd_resp_fraction: np.ndarray

    def __post_init__(self):
        for name in ("member_ids", "region_ids"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if self.allocation_mode not in ("fixed_calibration", "compensated"):
            raise ValueError("unknown allocation mode")
        shape = (len(self.member_ids), len(self.region_ids))
        for name in ("live_production", "live_mortality", "live_fire", "live_crop", "live_graz", "live_cwd_fraction"):
            object.__setattr__(self, name, _array(getattr(self, name), shape + (6,), name))
        for name in ("cwd_loss", "soil_environment_loss", "soil_input_exponent", "soil_input_cap", "fall_anchor", "cwd_resp_fraction"):
            object.__setattr__(self, name, _array(getattr(self, name), shape + (5,), name, name != "soil_input_exponent"))
        if np.any(self.live_crop + self.live_graz > 1) or np.any(self.live_cwd_fraction > 1) or np.any(self.cwd_resp_fraction > 1):
            raise ValueError("invalid routing fractions")
        if np.any(self.soil_input_cap <= 0):
            raise ValueError("input response caps must be positive")

    def soil_rate(self, reference_fall):
        fall = np.asarray(reference_fall)
        if fall.shape != self.fall_anchor.shape or not np.isfinite(fall).all() or np.any(fall < 0):
            raise ValueError("reference fall must match the prepared layout and be nonnegative")
        ratio = np.divide(fall, self.fall_anchor, out=np.ones_like(fall), where=self.fall_anchor > 0)
        response = safe_exp(self.soil_input_exponent * np.log(safe_ratio(ratio)), self.soil_input_cap)
        rate = self.soil_environment_loss * response
        if not np.isfinite(rate).all() or np.any(rate < 0):
            raise ValueError("invalid prepared soil rate")
        return rate


class ReferenceAdapter:
    """Validate labeled inputs once; prepare numerical rates outside the hot loop.

    The allocation policy is explicit. The main fixed policy uses modern
    calibration allocation forever; the alternative prepares a compensated
    allocation at each environment and has its own PI normalization anchor.
    """
    def __init__(self, parameters, calibration_environment, *, allocation_mode="fixed_calibration"):
        if allocation_mode not in ("fixed_calibration", "compensated"):
            raise ValueError("unknown allocation mode")
        self.p = parameters
        if tuple(parameters.bio_land.values) != BIOMES:
            raise ValueError("native biome order must match the reference interface")
        self.member_ids = _ids(tuple(int(x) for x in parameters.source_config.values), "source_config")
        self.region_ids = _ids(tuple(str(x) for x in parameters.reg_land.values), "reg_land")
        self.mode = allocation_mode
        self._check(calibration_environment)
        wood, turnover, mortality = xr.broadcast(parameters.p_wood, parameters.t_wood, parameters.v_mort)
        self.beta = xr.DataArray(mortality_partition(wood, turnover, mortality).beta_effective, dims=wood.dims, coords=wood.coords)
        self.alpha_cal = self._allocation(calibration_environment)
        pi = xr.Dataset({"D_CO2": 0., "D_Tl": 0., "r_Pl": 1.})
        alpha_pi = self.alpha_cal if self.mode == "fixed_calibration" else self._allocation(pi)
        self.anchor = preindustrial_fall_anchor(parameters, alpha_pi).reference_fall_pi
        self.template = parameters.v_mort.transpose("config", "reg_land", "bio_land")

    def _check(self, value):
        xr.align(self.p, value, join="exact", copy=False)
        if "config" in value.dims:
            if "source_config" not in value.coords or not np.array_equal(value.source_config, self.p.source_config):
                raise ValueError("environment must retain the exact original source_config identities")

    def _allocation(self, environment):
        loss = physical_loss_rates(environment, self.p.sel(bio_land="Forest", drop=True))
        mu, fire = loss.mortality_rate, loss.fire_rate
        if bool(((mu + fire) <= 0).any()):
            raise ValueError("compensated calibration allocation requires positive forest loss")
        return (self.beta.sel(bio_land="Forest", drop=True) * mu + self.p.p_wood.sel(bio_land="Forest", drop=True) * fire) / (mu + fire)

    def _numeric(self, value):
        return xr.broadcast(value, self.template)[0].transpose("config", "reg_land", "bio_land").values

    def prepare(self, midpoint_environment):
        """Use an environment evaluated at the temporal midpoint of the step."""
        self._check(midpoint_environment)
        p = self.p
        production = self._numeric(physical_production(midpoint_environment, p).production_input)
        loss = physical_loss_rates(midpoint_environment, p)
        mu, fire = self._numeric(loss.mortality_rate), self._numeric(loss.fire_rate)
        beta, wood = self._numeric(self.beta), self._numeric(p.p_wood)
        alpha = self.alpha_cal if self.mode == "fixed_calibration" else self._allocation(midpoint_environment)
        alpha = self._numeric(alpha)[..., 0]
        if np.any((wood[..., 0] <= 0) | (wood[..., 0] >= 1)):
            raise ValueError("forest wood fraction must lie strictly within (0,1)")
        prod = production[..., POOL_BIOME].copy()
        prod[..., 0] *= alpha
        prod[..., 1] *= 1-alpha
        mortality = mu[..., POOL_BIOME].copy()
        mortality[..., 0] *= beta[..., 0] / wood[..., 0]
        mortality[..., 1] *= (1-beta[..., 0]) / (1-wood[..., 0])
        route = beta[..., POOL_BIOME].copy()
        route[..., 0], route[..., 1] = 1., 0.
        crop, graz = self._numeric(p.p_charv)[..., POOL_BIOME], self._numeric(p.p_graz)[..., POOL_BIOME]
        if np.any(crop[..., :2] != 0) or np.any(graz[..., :2] != 0):
            raise ValueError("forest crop and grazing production removal must be zero")
        # At fall=anchor, the input response is exactly one. Retain the same
        # native environment factors; only the dynamic input factor
        # is evaluated numerically inside a reference step.
        decay = reference_decay_rates(midpoint_environment, p, self.anchor, self.anchor)
        return ReferenceCoefficients(self.member_ids, self.region_ids, self.mode,
            prod, mortality, fire[..., POOL_BIOME], crop, graz, route,
            self._numeric(decay.cwd_rate), self._numeric(decay.soil_rate),
            self._numeric(p.x_resp_fall), self._numeric(f_max(xr.where(p.v_resp > 0, p.v_resp, 1.))),
            self._numeric(self.anchor), self._numeric(p.p_cwd_resp))


def initial_reference(coefficients, *, positive=False, time=0.):
    """Zero or simple 1 Pg C/Mha start only on strictly positive-loss channels.

    Urban live/soil are zero in both starts. This is an initialization convention
    on a nonunique channel, not evidence that arbitrary Urban stocks disappear.
    Urban CWD is also zero because it can feed nondecaying Urban soil.
    """
    c = coefficients
    live = np.zeros_like(c.live_production)
    cwd, soil = np.zeros_like(c.cwd_loss), np.zeros_like(c.cwd_loss)
    if positive:
        live[c.live_mortality + c.live_fire > 0] = 1.
        cwd[c.cwd_loss > 0] = 1.
        soil[c.soil_environment_loss > 0] = 1.
        # Urban CWD would feed zero-loss Urban soil and retain memory. It is
        # outside the dissipative support of this two-start convergence test.
        cwd[..., BIOMES.index("Urban")] = 0.
    return ReferenceState(c.member_ids, c.region_ids, time, live, cwd, soil)


@dataclass(frozen=True)
class ReferenceStep:
    state: ReferenceState
    reference_fall: np.ndarray
    cwd_rate: np.ndarray
    soil_rate: np.ndarray
    ledger: dict


def _by_biome(pools):
    return np.concatenate((pools[..., :2].sum(axis=-1, keepdims=True), pools[..., 2:]), axis=-1)


def advance_reference(state, coefficients, dt):
    """Second-order live/dead coupling with one integrated physical ledger.

    Freeze midpoint-environment coefficients, advance live exactly, and use its
    integrated fine litter/dt to set the shared reference soil rate. CWD and soil
    then receive the same integrated live transfers uniformly during the step.
    This is a second-order approximation, not an exact joint nonlinear solve.
    At dt=0 use instantaneous reference fall; every integrated flux stays zero.
    """
    c = coefficients
    if state.member_ids != c.member_ids or state.region_ids != c.region_ids:
        raise ValueError("state and prepared coefficients have different member/region identities")
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    live = advance_live_carbon(state.live, c.live_production, c.live_mortality, c.live_fire, dt,
                              p_charv=c.live_crop, p_graz=c.live_graz)
    cwd_input = _by_biome(live.mortality * c.live_cwd_fraction)
    fine_input = _by_biome(live.mortality * (1-c.live_cwd_fraction))
    if dt > 0:
        cwd_input_rate, fall = cwd_input / dt, fine_input / dt
    else:
        cwd_input_rate = _by_biome(c.live_mortality * state.live * c.live_cwd_fraction)
        fall = _by_biome(c.live_mortality * state.live * (1-c.live_cwd_fraction))
    soil_rate = c.soil_rate(fall)
    m, r, b = state.cwd.shape
    shape = (m, r*b)
    cells = tuple((region, biome) for region in c.region_ids for biome in BIOMES)
    zero_product = np.zeros(shape + (1,))
    rates = DeadCarbonRates(c.member_ids, cells, ("zero_reference_product",),
                           c.cwd_loss.reshape(shape), soil_rate.reshape(shape),
                           c.cwd_resp_fraction.reshape(shape), zero_product)
    dead = advance_dead_carbon(state.cwd.reshape(shape), state.soil.reshape(shape), zero_product,
                              cwd_input_rate.reshape(shape), fall.reshape(shape), zero_product, rates, dt,
                              member_ids=c.member_ids, biome_ids=cells, product_ids=rates.product_ids)
    new = ReferenceState(c.member_ids, c.region_ids, state.time + dt, live.carbon,
                         dead.cwd.reshape(m, r, b), dead.soil.reshape(m, r, b))
    ledger = {"production": _by_biome(live.production), "fire": _by_biome(live.fire),
              "crop_harvest": _by_biome(live.crop_harvest), "grazing": _by_biome(live.grazing),
              "live_to_cwd": cwd_input, "live_to_soil": fine_input,
              "cwd_to_soil": dead.cwd_to_soil.reshape(m, r, b),
              "cwd_respiration": dead.cwd_respiration.reshape(m, r, b),
              "soil_respiration": dead.soil_respiration.reshape(m, r, b)}
    ledger["atmospheric_loss"] = sum(ledger[k] for k in ("fire", "crop_harvest", "grazing", "cwd_respiration", "soil_respiration"))
    ledger["budget_residual"] = new.carbon_by_biome - state.carbon_by_biome - ledger["production"] + ledger["atmospheric_loss"]
    return ReferenceStep(new, fall, c.cwd_loss, soil_rate, ledger)

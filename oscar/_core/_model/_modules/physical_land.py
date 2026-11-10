"""Physical land primitives, independent of Model, area accounting and age kernels.

Environmental adapters accept xarray datasets with native OSCAR parameter names.
Their production and loss rates are densities (PgC/Mha/yr) and rates (1/yr).
The live-carbon integrator accepts TOTAL carbon and production (PgC, PgC/yr).
Callers own area conversion, age multipliers, dimension ordering and event fluxes.
Original OSCAR processes and parameter datasets are never modified here.
"""

from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

from .OSCAR_landC_density import Eq__r_npp, Eq__r_vmort, Eq__r_vfire
from ..._base.fct_solve import safe_exp
from .forest_age import _member_ids, _carbon_array, _scalar, align_members


def _require(condition, message):
    if not bool(np.all(np.asarray(condition))):
        raise ValueError(message)


def _finite_nonnegative(value, name):
    _require(np.isfinite(value) & (value >= 0), name + " must be finite and nonnegative")


def _co2_factor(co2, par):
    _require(np.isfinite(co2) & (co2 > 0), "Absolute CO2 must be finite and positive")
    _require(np.isfinite(par.CO2_piL) & (par.CO2_piL > 0), "CO2_piL must be positive")
    _require(np.isfinite(par.b_npp_CO2) & np.isfinite(par.x_npp_CO2), "Invalid CO2 response parameters")
    log_ratio = np.log(co2 / par.CO2_piL)
    x = par.x_npp_CO2
    # The continuous x=0 limit is log(C/CL); no zero division is evaluated.
    denominator = xr.where(x == 0, 1.0, x)
    response = xr.where(x == 0, log_ratio, np.expm1(x * log_ratio) / denominator)
    return 1.0 + par.b_npp_CO2 * response


def physical_production(environment, parameters):
    """Return signed fitted production, physical input and boundary correction.

    Values are PgC/Mha/yr. The nonnegative input boundary belongs to this
    external-input state model; it is not a general ecological ban on negative
    NPP. Temperature and precipitation retain native OSCAR nonlinear factors.
    """
    e, p = environment, parameters
    _finite_nonnegative(p.k_npp, "k_npp")
    _finite_nonnegative(p.npp_piL, "npp_piL")
    _require(np.isfinite(e.D_Tl), "D_Tl must be finite")
    _require(np.isfinite(e.r_Pl) & (e.r_Pl > 0), "r_Pl must be finite and positive")
    co2 = _co2_factor(p.CO2_pi + e.D_CO2, p)
    temperature = safe_exp(p.g_npp_T * e.D_Tl, 100) * np.exp(
        -p.g_npp_T2 * np.maximum(e.D_Tl, 0) ** 2
    )
    precipitation = safe_exp(p.x_npp_P * np.log(e.r_Pl), 100)
    raw = p.k_npp * p.npp_piL * co2 * temperature * precipitation
    _require(np.isfinite(raw), "Nonfinite fitted production")
    physical = xr.where(raw >= 0, raw, 0.0)
    result = xr.Dataset({
        "production_raw": raw,
        "production_input": physical,
        "production_correction": physical - raw,
        "production_boundary_active": raw < 0,
    })
    for name in ("production_raw", "production_input", "production_correction"):
        result[name].attrs["units"] = "PgC Mha-1 yr-1"
    return result


def physical_loss_rates(environment, parameters):
    """Native pure-environment mortality/fire; never pass age-modified NPP.

    A nonpositive PI CO2 normalization is admissible only if BOTH NPP feedback
    exponents are zero. Such cells bypass the invalid normalization before any
    native response is evaluated. Their production factor for losses is one.
    The affected cells in the supplied ensemble have both exponents zero. Future incompatible
    parameter sets fail explicitly instead of acquiring a new anchor or epsilon.
    """
    p = parameters
    _finite_nonnegative(p.v_mort, "v_mort")
    _finite_nonnegative(p.v_fire, "v_fire")
    _require(np.isfinite(environment.D_Tl), "D_Tl must be finite")
    _require(np.isfinite(environment.r_Pl) & (environment.r_Pl > 0), "r_Pl must be positive")
    _co2_factor(p.CO2_pi + environment.D_CO2, p)
    invalid_reference = _co2_factor(p.CO2_pi, p) <= 0
    _require(
        (~invalid_reference) | ((p.x_mort_npp == 0) & (p.x_fire_npp == 0)),
        "Nonpositive PI production normalization with nonzero NPP loss feedback",
    )
    # Assignment replaces variables on a shallow dataset copy, never the source.
    safe = p.copy(deep=False)
    safe["b2_npp_CO2"] = xr.where(invalid_reference, 0.0, p.b2_npp_CO2)
    safe["x_npp_CO2"] = xr.where(invalid_reference, 1.0, p.x_npp_CO2)
    # Native f_max divides by the base rate. Zero-rate channels cannot carry
    # flux; evaluate their irrelevant response with a finite dummy rate first.
    safe["v_mort"] = xr.where(p.v_mort > 0, p.v_mort, 1.0)
    safe["v_fire"] = xr.where(p.v_fire > 0, p.v_fire, 1.0)
    ratio = xr.where(invalid_reference, 1.0, Eq__r_npp(environment, safe))
    _require(np.isfinite(ratio) & (ratio > 0), "Invalid native loss response input")
    variables = environment.assign(r_npp=ratio)
    mortality = p.v_mort * Eq__r_vmort(variables, safe)
    fire = p.v_fire * Eq__r_vfire(variables, safe)
    _finite_nonnegative(mortality, "mortality_rate")
    _finite_nonnegative(fire, "fire_rate")
    result = xr.Dataset({
        "mortality_rate": mortality,
        "fire_rate": fire,
        "r_npp_for_loss": ratio,
        "invalid_PI_normalization": invalid_reference,
    })
    result.mortality_rate.attrs["units"] = "yr-1"
    result.fire_rate.attrs["units"] = "yr-1"
    return result


@dataclass(frozen=True)
class MortalityPartition:
    beta_raw: np.ndarray
    beta_effective: np.ndarray
    boundary_active: np.ndarray


def mortality_partition(p_wood, t_wood, v_mort, *, routing="bounded"):
    """Broadcast parameters to a conservative, fixed mortality allocation.

    routing='fine_endpoint' changes ONLY raw beta>1 cells to beta=0. It is the
    registered extreme routing sensitivity, not a parameter replacement.
    Zero mortality and zero wood channel imply beta=0; a positive wood channel
    with zero mortality is incompatible and raises.
    """
    wood, turnover, mortality = np.broadcast_arrays(
        np.asarray(p_wood, dtype=float), np.asarray(t_wood, dtype=float),
        np.asarray(v_mort, dtype=float),
    )
    _require(np.isfinite(wood) & (wood >= 0) & (wood <= 1), "p_wood must be in [0,1]")
    _require(np.isfinite(turnover) & (turnover > 0), "t_wood must be finite and positive")
    _finite_nonnegative(mortality, "v_mort")
    channel = wood / turnover
    _require((mortality > 0) | (channel == 0), "Positive wood channel with zero mortality")
    raw = np.divide(channel, mortality, out=np.zeros_like(channel), where=mortality > 0)
    _require(np.isfinite(raw), "Nonfinite beta")
    if routing == "bounded":
        effective = np.minimum(raw, 1.0)
    elif routing == "fine_endpoint":
        effective = np.where(raw > 1, 0.0, raw)
    else:
        raise ValueError("Unknown mortality routing: " + str(routing))
    return MortalityPartition(raw, effective, raw > 1)


def route_mortality(mortality_carbon, beta_effective):
    """Return CWD and fine-litter carbon from the SAME integrated mortality."""
    mortality, beta = np.broadcast_arrays(
        np.asarray(mortality_carbon, dtype=float), np.asarray(beta_effective, dtype=float)
    )
    _finite_nonnegative(mortality, "mortality_carbon")
    _require(np.isfinite(beta) & (beta >= 0) & (beta <= 1), "beta_effective must be in [0,1]")
    cwd = beta * mortality
    return cwd, mortality - cwd


@dataclass(frozen=True)
class LiveCarbonStep:
    carbon: np.ndarray
    production: np.ndarray
    effective_input: np.ndarray
    crop_harvest: np.ndarray
    grazing: np.ndarray
    mortality: np.ndarray
    fire: np.ndarray
    integrated_carbon: np.ndarray


def advance_live_carbon(carbon, production_rate, mortality_rate, fire_rate, dt,
                        *, p_charv=0.0, p_graz=0.0):
    """Exact constant-coefficient step, with all integrated fluxes in PgC.

    carbon is PgC; production_rate is TOTAL physical production in PgC/yr,
    already multiplied by area and any caller-provided age factor. dt is yr;
    mortality_rate/fire_rate are 1/yr. Fractions act on
    production before harvest and grazing. No events, soil pools or area conversion are hidden here.
    All arguments follow NumPy broadcasting; for (500, cohort) states use
    (500, 1) member-specific rates, never a bare (500,) axis.
    """
    c, production_rate, m, f, dt, crop, graz = np.broadcast_arrays(*[
        np.asarray(value, dtype=float) for value in
        (carbon, production_rate, mortality_rate, fire_rate, dt, p_charv, p_graz)
    ])
    for value, name in ((c, "carbon"), (production_rate, "production_rate"),
                        (m, "mortality_rate"), (f, "fire_rate"), (dt, "dt"),
                        (crop, "p_charv"), (graz, "p_graz")):
        _finite_nonnegative(value, name)
    _require(crop + graz <= 1, "p_charv + p_graz must not exceed one")
    effective_rate = production_rate * (1.0 - (crop + graz))
    z = (m + f) * dt
    _require(np.isfinite(z), "Nonfinite total loss-rate times dt")
    phi1 = np.divide(-np.expm1(-z), z, out=np.ones_like(z), where=z != 0)
    phi2 = np.divide(1.0 - phi1, z, out=np.full_like(z, 0.5), where=z != 0)
    small = z < 1e-4
    # Series for (z - 1 + exp(-z))/z**2 avoids cancellation near zero.
    zs = np.where(small, z, 0.0)
    series = 0.5 + zs * (-1/6 + zs * (1/24 + zs * (-1/120 + zs/720)))
    phi2 = np.where(small, series, phi2)
    a = dt * phi1
    integral = c * a + effective_rate * dt * dt * phi2
    new = c * np.exp(-z) + effective_rate * a
    production = production_rate * dt
    result = LiveCarbonStep(
        new, production, effective_rate * dt, crop * production, graz * production,
        m * integral, f * integral, integral,
    )
    for value in result.__dict__.values():
        _finite_nonnegative(value, "Integrated carbon or flux")
    return result


# Isolated two-pool candidate. No Model registration or soil/HWP decay here.


@dataclass(frozen=True)
class TwoPoolRates:
    member_ids: tuple
    mortality: np.ndarray  # (member, W/F), yr-1
    fire: np.ndarray  # (member,), yr-1
    allocation: np.ndarray  # (member, W/F), production shares

    def __post_init__(self):
        ids = _member_ids(self.member_ids)
        object.__setattr__(self, "member_ids", ids)
        for name, shape in (("mortality", (len(ids), 2)), ("fire", (len(ids),)),
                            ("allocation", (len(ids), 2))):
            object.__setattr__(self, name, _carbon_array(getattr(self, name), shape, name))
        _require(np.all(self.allocation <= 1) & np.all(self.allocation.sum(axis=1) == 1),
                 "production allocations must sum to one")

    @property
    def loss(self):
        return self.mortality + self.fire[:, None]

    @property
    def unique_equilibrium(self):
        return np.all(self.loss > 0, axis=1)


def two_pool_rates(p_wood, beta, mortality_rate, fire_rate, *, member_ids,
                   allocation_wood=None):
    """Pool loss priors and explicit or dynamically compensated production shares.

    Working-model callers must pass the fixed Ecal allocation_wood vector in
    member_ids order. Omitting it retains the dynamic compensation sensitivity
    alpha=(beta*mu+p*fire)/(mu+fire), not an implicit working-model choice.
    p is reference composition; mu/beta scale the pool loss priors, not actual
    whole-stock mortality or routing fractions at arbitrary composition.

    With explicit alpha and positive pool losses, mature densities are
    alpha*N/lambda_w and (1-alpha)*N/lambda_f. They need not preserve p or
    total N/(mu+fire) outside Ecal. Zero loss permits linear accumulation with
    positive input, or initial-stock memory with zero input. Zero total loss
    is rejected only when alpha must be inferred from the dynamic formula.
    """
    ids = _member_ids(member_ids)
    values = []
    for value, name in ((p_wood, "p_wood"), (beta, "beta"),
                        (mortality_rate, "mortality"), (fire_rate, "fire")):
        value = _carbon_array(value, (len(ids),), name)
        values.append(value)
    p, beta, mu, fire = values
    _require((p > 0) & (p < 1), "two-pool p_wood must be strictly between zero and one")
    _require(beta <= 1, "beta must be in [0,1]")
    mortality = np.stack((beta * mu / p, (1 - beta) * mu / (1 - p)), axis=1)
    if allocation_wood is None:
        _require(mu + fire > 0, "zero total loss has undefined production allocation")
        # Cancel p analytically: beta=1 and fire=0 must give alpha exactly one.
        alpha = (beta * mu + p * fire) / (mu + fire)
    else:
        alpha = _carbon_array(allocation_wood, (len(ids),), "allocation_wood")
        _require(alpha <= 1, "allocation_wood must be in [0,1]")
    allocation = np.stack((alpha, 1 - alpha), axis=1)
    _require(np.isfinite(mortality) & (mortality >= 0), "invalid pool mortality")
    _require(np.isfinite(allocation) & (allocation >= 0) & (allocation <= 1), "invalid allocation")
    return TwoPoolRates(ids, mortality, fire, allocation)



@dataclass(frozen=True)
class TwoPoolStep:
    carbon: np.ndarray
    production: np.ndarray
    mortality: np.ndarray
    fire: np.ndarray
    integrated_carbon: np.ndarray

    @property
    def cwd(self):
        return self.mortality[:, 0, :]

    @property
    def soil(self):
        return self.mortality[:, 1, :]

    @property
    def fire_total(self):
        return self.fire.sum(axis=1)


def advance_two_pool_carbon(carbon, production_rate, rates, dt, *, member_ids):
    """Constant-coefficient step for exact (member, W/F, cohort/source) arrays.

    Production is TOTAL input, already area/age/allocation adjusted. Changing
    coefficients or age inputs must be sampled by the caller (midpoint in the
    candidate trials). Frozen-coefficient exactness is not variable-step exactness.
    """
    dt = _scalar(dt, "dt")
    ids = _member_ids(member_ids)
    carbon = np.asarray(carbon, dtype=np.float64)
    if carbon.ndim != 3 or carbon.shape[:2] != (len(ids), 2):
        raise ValueError("carbon must have shape (member, 2, cohort)")
    carbon = _carbon_array(carbon, carbon.shape, "carbon")
    production = _carbon_array(production_rate, carbon.shape, "production_rate")
    mortality = align_members(rates.mortality, rates.member_ids, ids)[:, :, None]
    fire = align_members(rates.fire, rates.member_ids, ids)[:, None, None]
    step = advance_live_carbon(carbon, production, mortality, fire, dt)
    return TwoPoolStep(step.carbon, step.production, step.mortality,
                       step.fire, step.integrated_carbon)


def advance_two_pool_forest(state, npp_density, secondary_factors, rates, dt, *, member_ids):
    """Advance common-area forest; primary uses mature production, not an age.

    npp_density has (member,), factors have (member, W/F, secondary cohort).
    These arrays share explicit member_ids. Returned sources include primary
    first, enabling each integrated transfer to be booked once by the caller.
    """
    npp = align_members(npp_density, member_ids, state.member_ids)
    npp = _carbon_array(npp, (len(state.member_ids),), "mature production")
    factors = align_members(secondary_factors, member_ids, state.member_ids)
    factors = _carbon_array(factors, state.secondary_carbon.shape, "age factors")
    factors = np.concatenate((np.ones_like(state.primary_carbon)[:, :, None], factors), axis=2)
    allocation = align_members(rates.allocation, rates.member_ids, state.member_ids)
    area = np.concatenate(([state.primary_area], state.secondary_area))
    production = npp[:, None, None] * allocation[:, :, None] * factors * area[None, None, :]
    carbon = np.concatenate((state.primary_carbon[:, :, None], state.secondary_carbon), axis=2)
    step = advance_two_pool_carbon(carbon, production, rates, dt, member_ids=state.member_ids)
    time = state.time + float(dt)
    if not np.isfinite(time) or (dt > 0 and time <= state.time):
        raise ValueError("time increment is not representable")
    return replace(state, time=time, primary_carbon=step.carbon[:, :, 0],
                   secondary_carbon=step.carbon[:, :, 1:]), step


@dataclass(frozen=True)
class TwoPoolHarvestRouting:
    cwd: np.ndarray  # (member, source)
    soil: np.ndarray  # (member, source)
    hwp: np.ndarray  # (member, source, product)


def route_two_pool_harvest(removal, p_hwp, *, member_ids):
    """Product shares apply only to removed actual W; all actual F goes to soil."""
    shares = align_members(p_hwp, member_ids, removal.member_ids)
    if shares.ndim != 2 or shares.shape[1] == 0:
        raise ValueError("p_hwp must have shape (member, product)")
    _finite_nonnegative(shares, "p_hwp")
    total = shares.sum(axis=1)
    _require(total <= 1, "HWP shares exceed available wood")
    wood, other = removal.carbon[:, 0, :], removal.carbon[:, 1, :]
    return TwoPoolHarvestRouting(wood * (1 - total[:, None]), other.copy(),
                                 wood[:, :, None] * shares[:, None, :])

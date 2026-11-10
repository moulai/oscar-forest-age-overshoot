"""One-region actual land carbon coupling with independent reference decay rates.

Stocks are Pg C, areas Mha, time years. Prepared production is Pg C/Mha/year.
Events act separately at step boundaries on the returned LandEventState.
"""
from dataclasses import dataclass, replace

import numpy as np

from .dead_carbon import DeadCarbonRates, advance_dead_carbon
from .dynamic_reference import BIOMES
from .forest_age import _carbon_array, _member_ids, _scalar, align_members
from .growth_curve import GrowthCurve
from .land_events import LandEventState
from .physical_land import advance_live_carbon

LIVE_FIELDS = ("live_production", "live_mortality", "live_fire",
               "live_crop", "live_graz", "live_cwd_fraction")


@dataclass(frozen=True)
class AgeProduction:
    """Fixed calibration denominator; environmental losses never replace it."""
    curve: GrowthCurve
    region_id: str
    member_ids: tuple
    lambda_w_cal: np.ndarray
    tau_years: float = 2.
    slow_fraction: float = 0.
    time_scale: float = 1.

    def __post_init__(self):
        ids = _member_ids(self.member_ids)
        object.__setattr__(self, "member_ids", ids)
        lam = _carbon_array(self.lambda_w_cal, (len(ids),), "lambda_w_cal")
        if np.any(lam <= 0):
            raise ValueError("lambda_w_cal must be strictly positive")
        if self.region_id not in self.curve.region_ids:
            raise ValueError("age-production region is absent from the curve")
        object.__setattr__(self, "lambda_w_cal", lam)
        for name in ("tau_years", "slow_fraction", "time_scale"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if self.slow_fraction > 1 or self.time_scale == 0:
            raise ValueError("invalid slow fraction or time scale")

    @classmethod
    def from_calibration(cls, curve, calibration, region_id, **kwargs):
        """Prepare an accepted labeled calibration outside the integration loop."""
        if calibration.source_config.dims != ("config",):
            raise ValueError("calibration must retain source_config on config")
        if set(map(str, calibration.reg_land.values)) != set(curve.region_ids):
            raise ValueError("calibration and curve regions differ")
        selected = calibration.sel(reg_land=region_id)
        return cls(curve, region_id, tuple(selected.source_config.values.tolist()),
                   selected.lambda_w_cal.transpose("config").values, **kwargs)

    def factors(self, age, *, member_ids):
        """Return (member, W/F, cohort) production multipliers at supplied ages."""
        age = np.asarray(age, dtype=float)
        if age.ndim != 1:
            raise ValueError("cohort ages must be a vector")
        s, slope = self.curve.evaluate(age / self.time_scale, regions=(self.region_id,))
        s, slope = s[0], slope[0] / self.time_scale
        lam = align_members(self.lambda_w_cal, self.member_ids, member_ids)
        wood = s[None, :] + slope[None, :] / lam[:, None]
        h = np.ones_like(age) if self.tau_years == 0 else -np.expm1(-age / self.tau_years)
        fine = (1-self.slow_fraction)*h + self.slow_fraction*s
        return np.stack((wood, np.broadcast_to(fine, wood.shape)), axis=1)


@dataclass(frozen=True)
class LandCarbonCoefficients:
    """Region slice of prepared reference inputs plus externally supplied decay.

    The dead rates must come from an independently advanced reference and an
    explicit product-decay source. No actual forest litter feeds back into them.
    """
    member_ids: tuple
    region_id: str
    live_production: np.ndarray
    live_mortality: np.ndarray
    live_fire: np.ndarray
    live_crop: np.ndarray
    live_graz: np.ndarray
    live_cwd_fraction: np.ndarray
    dead_rates: DeadCarbonRates
    allocation_mode: str = "fixed_calibration"

    def __post_init__(self):
        ids = _member_ids(self.member_ids)
        object.__setattr__(self, "member_ids", ids)
        if not isinstance(self.region_id, str) or not self.region_id:
            raise ValueError("region_id must be nonempty")
        if self.allocation_mode not in ("fixed_calibration", "compensated"):
            raise ValueError("unknown allocation mode")
        for name in LIVE_FIELDS:
            object.__setattr__(self, name, _carbon_array(getattr(self, name), (len(ids), 6), name))
        if (np.any(self.live_crop + self.live_graz > 1)
                or np.any(self.live_cwd_fraction > 1)):
            raise ValueError("invalid production-removal or mortality-routing fraction")
        if (np.any(self.live_crop[:, :2] != 0) or np.any(self.live_graz[:, :2] != 0)
                or np.any(self.live_cwd_fraction[:, 0] != 1)
                or np.any(self.live_cwd_fraction[:, 1] != 0)):
            raise ValueError("forest W mortality must enter CWD and F mortality must enter soil")
        if self.dead_rates.biome_ids != BIOMES:
            raise ValueError("dead rates must use native biome order")
        align_members(self.dead_rates.cwd_loss, self.dead_rates.member_ids, ids)

    @classmethod
    def from_reference(cls, coefficients, region_id, dead_rates):
        """Preserve already allocated production, including fixed alpha_cal."""
        ri = coefficients.region_ids.index(region_id)
        return cls(coefficients.member_ids, region_id,
                   *(getattr(coefficients, name)[:, ri, :] for name in LIVE_FIELDS),
                   dead_rates, coefficients.allocation_mode)


def carbon_by_biome(state):
    """Actual stocks by native biome, including products on their source biome."""
    out = state.cwd + state.soil + state.hwp.sum(axis=2)
    out = out.copy()
    out[:, 0] += state.forest.total_carbon
    classes = {x.class_id: x for x in state.classes}
    for j, name in enumerate(state.nonforest_classes):
        out[:, state.biome_ids.index(classes[name].biome)] += state.nonforest_carbon[:, j]
    return out


@dataclass(frozen=True)
class LandCarbonStep:
    state: LandEventState
    ledger: dict
    forest_live_carbon_time_integral: np.ndarray  # member, W/F; Pg C yr, including primary


def advance_land_carbon(state, coefficients, dt, *, age_production):
    """Advance actual stocks and all integrated fluxes without executing events.

    Primary production is mature. Secondary age inputs and environmental
    coefficients are sampled at the temporal midpoint. Mortality is integrated
    by the live solver, aggregated by biome once, then supplied uniformly to the
    dead solver. This coupling is second order, not an exact joint ODE solve.
    None for age_production is the explicit no-age structural control.
    """
    c = coefficients
    dt = _scalar(dt, "dt")
    ids = state.forest.member_ids
    if state.forest.region_id != c.region_id or state.biome_ids != BIOMES:
        raise ValueError("state region or biome coordinates differ from coefficients")
    if state.product_ids != c.dead_rates.product_ids:
        raise ValueError("state product coordinates differ from decay rates")
    if any(x.biome != "Forest" for x in state.classes if x.role != "nonforest"):
        raise ValueError("forest classes must use the native Forest biome")
    fields = [align_members(getattr(c, name), c.member_ids, ids) for name in LIVE_FIELDS]
    prod, mortality, fire, crop, graz, route = fields
    factors = np.ones_like(state.forest.secondary_carbon)
    if age_production is not None:
        if age_production.region_id != state.forest.region_id:
            raise ValueError("age-production and state regions differ")
        factors = age_production.factors(
            state.forest.time + dt/2 - state.forest.birth_time, member_ids=ids)
    factors = np.concatenate((np.ones_like(state.forest.primary_carbon)[:, :, None], factors), axis=2)
    area = np.r_[state.forest.primary_area, state.forest.secondary_area]
    forest_stock = np.concatenate((state.forest.primary_carbon[:, :, None],
                                   state.forest.secondary_carbon), axis=2)
    forest = advance_live_carbon(forest_stock, prod[:, :2, None]*factors*area[None, None, :],
                                mortality[:, :2, None], fire[:, :2, None], dt)
    classes = {x.class_id: x for x in state.classes}
    nf_biomes = np.array([BIOMES.index(classes[x].biome) for x in state.nonforest_classes], dtype=int)
    nf_pools = nf_biomes + 1
    nonforest = advance_live_carbon(
        state.nonforest_carbon, prod[:, nf_pools]*state.nonforest_area[None, :],
        mortality[:, nf_pools], fire[:, nf_pools], dt,
        p_charv=crop[:, nf_pools], p_graz=graz[:, nf_pools])

    def aggregate(forest_value, nonforest_value):
        out = np.zeros((len(ids), len(BIOMES)))
        out[:, 0] = forest_value.sum(axis=(1, 2))
        for j, bi in enumerate(nf_biomes):
            out[:, bi] += nonforest_value[:, j]
        return out

    ledger = {key: aggregate(getattr(forest, key), getattr(nonforest, key))
              for key in ("production", "fire", "crop_harvest", "grazing")}
    cwd_input = aggregate(forest.mortality*route[:, :2, None],
                          nonforest.mortality*route[:, nf_pools])
    soil_input = aggregate(forest.mortality*(1-route[:, :2, None]),
                           nonforest.mortality*(1-route[:, nf_pools]))
    # dt=0 integrates no transfer and avoids dividing by zero.
    zeros = np.zeros_like(state.hwp)
    dead = advance_dead_carbon(
        state.cwd, state.soil, state.hwp,
        cwd_input/dt if dt else np.zeros_like(cwd_input),
        soil_input/dt if dt else np.zeros_like(soil_input), zeros, c.dead_rates, dt,
        member_ids=ids, biome_ids=state.biome_ids, product_ids=state.product_ids)
    time = state.forest.time + dt
    if not np.isfinite(time) or (dt > 0 and time <= state.forest.time):
        raise ValueError("time increment is not representable")
    new_forest = replace(state.forest, time=time, primary_carbon=forest.carbon[:, :, 0],
                         secondary_carbon=forest.carbon[:, :, 1:])
    new = replace(state, forest=new_forest, nonforest_carbon=nonforest.carbon,
                  cwd=dead.cwd, soil=dead.soil, hwp=dead.hwp)
    ledger.update(live_to_cwd=cwd_input, live_to_soil=soil_input,
                  cwd_to_soil=dead.cwd_to_soil, cwd_respiration=dead.cwd_respiration,
                  soil_respiration=dead.soil_respiration, hwp_respiration=dead.hwp_respiration)
    ledger["atmospheric_loss"] = sum(ledger[k] for k in
        ("fire", "crop_harvest", "grazing", "cwd_respiration", "soil_respiration")) + dead.hwp_respiration.sum(axis=2)
    ledger["budget_residual"] = carbon_by_biome(new)-carbon_by_biome(state)-ledger["production"]+ledger["atmospheric_loss"]
    if np.any(abs(ledger["budget_residual"]) > 1e-10*np.maximum(1., carbon_by_biome(state))):
        raise ValueError("actual carbon budget failed")
    return LandCarbonStep(new, ledger, forest.integrated_carbon.sum(axis=2))

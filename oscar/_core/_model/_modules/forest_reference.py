"""Pure physical reference normalization and dead-carbon environmental rates.

No actual age-affected litter input, initial state assignment, spin-up, land
conversion or Model registration occurs here. Density fluxes are PgC/Mha/yr.
"""
import numpy as np
import xarray as xr

from ..._base.fct_solve import safe_exp, safe_ratio, f_max
from .OSCAR_landC_density import Eq__r_vcwd
from .physical_land import physical_production, physical_loss_rates, mortality_partition


def _require(condition, message):
    if not bool(np.all(np.asarray(condition))):
        raise ValueError(message)


def _nonnegative(value, name):
    _require(np.isfinite(value) & (value >= 0), name + " must be finite and nonnegative")


def _aligned(*objects):
    objects = xr.align(*objects, join="exact", copy=False)
    identities = [obj.source_config for obj in objects if "source_config" in obj.coords]
    if identities:
        for obj in objects:
            if "config" in obj.dims and "source_config" not in obj.coords:
                raise ValueError("config-bearing inputs require source_config when an authoritative identity is present")
        for other in identities[1:]:
            _require(np.array_equal(identities[0].values, other.values), "source_config identities differ")
    return objects


def mature_reference_fall(production, mortality, fire, beta, *,
                          wood_fraction=None, wood_allocation=None, p_charv=0., p_graz=0.):
    """An equilibrium FLUX identity for normalization, never an initial stock.

    Forest W/F allocations are explicit. The fixed-calibration main case passes
    alpha_cal; a dynamic-allocation sensitivity passes alpha of this environment.
    Nonforest input first loses the declared crop/grazing production fractions.
    Zero loss with positive input has no finite mature reference and raises.
    Zero input with zero loss has zero fall but a nonunique unassigned stock.
    """
    # Enforce identities even when this flux helper is called outside the PI wrapper.
    _aligned(*[value for value in (production, mortality, fire, beta, wood_fraction,
                                   wood_allocation, p_charv, p_graz) if isinstance(value, xr.DataArray)])
    for value, name in ((production, "production"), (mortality, "mortality"),
                        (fire, "fire"), (beta, "beta"), (p_charv, "p_charv"), (p_graz, "p_graz")):
        _nonnegative(value, name)
    _require(beta <= 1, "beta must not exceed one")
    _require(p_charv + p_graz <= 1, "crop plus grazing fractions exceed one")
    if wood_fraction is None:
        if wood_allocation is not None:
            raise ValueError("wood allocation requires forest wood fraction")
        inputs = [production * (1-p_charv-p_graz)]
        losses = [mortality+fire]
        fine_mortality = (1-beta)*mortality
    else:
        _require(wood_allocation is not None, "forest allocation must be explicit")
        _require(np.isfinite(wood_fraction) & (wood_fraction > 0) & (wood_fraction < 1), "forest wood fraction must be in (0,1)")
        _require(np.isfinite(wood_allocation) & (wood_allocation >= 0) & (wood_allocation <= 1), "forest allocation must be in [0,1]")
        _require((p_charv == 0) & (p_graz == 0), "forest reference does not support crop/grazing production removal")
        inputs = [production*wood_allocation, production*(1-wood_allocation)]
        fine_mortality = (1-beta)*mortality/(1-wood_fraction)
        losses = [beta*mortality/wood_fraction+fire, fine_mortality+fire]
    nonunique = xr.zeros_like(production, dtype=bool)
    for input_rate, loss in zip(inputs, losses):
        _nonnegative(loss, "pool loss")
        _require((loss > 0) | (input_rate == 0), "positive input with zero loss has no finite reference anchor")
        nonunique = nonunique | (loss == 0)
    # Loss is replaced only in an inactive zero/zero normalization denominator.
    denominator = xr.where(losses[-1] > 0, losses[-1], 1.)
    fall = inputs[-1] * (fine_mortality / denominator)
    _nonnegative(fall, "mature reference fall")
    return xr.Dataset({"reference_fall": fall, "zero_loss_nonunique": nonunique})


def preindustrial_fall_anchor(parameters, forest_allocation):
    """New PI normalization for all biomes under the selected forest allocation.

    The caller must pass the allocation actually used by the PI reference:
    alpha_cal for the fixed-allocation main case, alpha_PI for the compensated
    sensitivity. Old Par.ffall_pi is never read. No equilibrium stock is returned.
    """
    p, allocation = _aligned(parameters, forest_allocation)
    _require("bio_land" in p.dims and "Forest" in p.bio_land.values, "expected native biomes including Forest")
    pi = xr.Dataset({"D_CO2": 0., "D_Tl": 0., "r_Pl": 1.})
    production = physical_production(pi, p).production_input
    rates = physical_loss_rates(pi, p)
    wood, turnover, mortality = xr.broadcast(p.p_wood, p.t_wood, p.v_mort)
    partition = mortality_partition(wood.values, turnover.values, mortality.values)
    beta = xr.DataArray(partition.beta_effective, dims=wood.dims, coords=wood.coords)
    rows = []
    for biome in p.bio_land.values:
        select = dict(bio_land=biome)
        kwargs = {"wood_fraction": p.p_wood.sel(**select, drop=True), "wood_allocation": allocation} if biome == "Forest" else {}
        result = mature_reference_fall(production.sel(**select, drop=True),
            rates.mortality_rate.sel(**select, drop=True), rates.fire_rate.sel(**select, drop=True),
            beta.sel(**select, drop=True), p_charv=p.p_charv.sel(**select, drop=True),
            p_graz=p.p_graz.sel(**select, drop=True), **kwargs)
        rows.append(result.expand_dims(bio_land=[biome]))
    result = xr.concat(rows, dim="bio_land").transpose(*p.v_mort.dims)
    result = result.rename(reference_fall="reference_fall_pi")
    result.reference_fall_pi.attrs["units"] = "PgC Mha-1 yr-1"
    result.attrs["role"] = "Normalization flux identity only; not initial stocks or completed dynamic spin-up"
    return result


def reference_decay_rates(environment, parameters, reference_fall, reference_fall_pi):
    """Native thermal/moisture responses with a new reference-only litter anchor.

    Actual age-affected litter must not be supplied. Positive anchors use the
    reference fall/PI fall ratio; a zero anchor has neutral input response while
    retaining temperature and water responses. True zero base rates remain zero;
    finite dummy bases are used only to evaluate mathematically irrelevant caps.
    """
    e, p, fall, anchor = _aligned(environment, parameters, reference_fall, reference_fall_pi)
    for value, name in ((fall,"reference_fall"), (anchor,"reference_fall_pi"),
                        (p.v_cwd,"v_cwd"), (p.v_resp,"v_resp")):
        _nonnegative(value, name)
    _require(np.isfinite(e.D_Tl), "D_Tl must be finite")
    _require(np.isfinite(e.r_Pl) & (e.r_Pl > 0), "r_Pl must be finite and positive")
    safe = p.copy(deep=False)
    safe["v_cwd"] = xr.where(p.v_cwd > 0, p.v_cwd, 1.)
    safe["v_resp"] = xr.where(p.v_resp > 0, p.v_resp, 1.)
    cwd_response = Eq__r_vcwd(e, safe)
    ratio = xr.where(anchor > 0, fall/xr.where(anchor > 0, anchor, 1.), 1.)
    cap = f_max(safe.v_resp)
    input_response = safe_exp(p.x_resp_fall * np.log(safe_ratio(ratio)), cap)
    temperature = safe_exp(p.g_resp_T * e.D_Tl, cap)
    precipitation = safe_exp(p.x_resp_P * np.log(e.r_Pl), cap)
    soil_response = input_response*temperature*precipitation
    cwd_rate = xr.broadcast(p.v_cwd*cwd_response, p.v_resp)[0]
    soil_rate = p.v_resp*soil_response
    for value, name in ((cwd_rate,"cwd_rate"), (soil_rate,"soil_rate"),
                        (input_response,"input_response"), (soil_response,"soil_response")):
        _nonnegative(value, name)
    result = xr.Dataset({"cwd_rate":cwd_rate, "soil_rate":soil_rate,
                         "soil_input_response":input_response, "soil_temperature_response":temperature,
                         "soil_precipitation_response":precipitation, "soil_response":soil_response,
                         "zero_reference_anchor":anchor == 0})
    for name in ("cwd_rate","soil_rate"):
        result[name].attrs["units"] = "yr-1"
    result.attrs["litter_source"] = "Physical reference trajectory only; actual age-affected litter excluded"
    return result

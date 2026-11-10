"""Actual CWD, soil and wood-product stocks, with one integrated transfer ledger.

This isolated linear step has no Model registration or land-use policy. Stocks
are Pg C, external rates Pg C yr-1, loss rates yr-1, and dt years. CWD/soil use
(member, biome); products use (member, biome, product). Soil and CWD are mixed
within biome, while products remain on their originating product ledger.
"""
from dataclasses import dataclass
import math

import numpy as np

from .forest_age import _member_ids, _carbon_array, _scalar, align_members
from .physical_land import advance_live_carbon


def _coordinates(values, name):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    return values


@dataclass(frozen=True)
class DeadCarbonRates:
    member_ids: tuple
    biome_ids: tuple
    product_ids: tuple
    cwd_loss: np.ndarray
    soil_loss: np.ndarray
    cwd_resp_fraction: np.ndarray
    hwp_loss: np.ndarray

    def __post_init__(self):
        for name, check in (("member_ids", _member_ids),
                            ("biome_ids", lambda v: _coordinates(v, "biome_ids")),
                            ("product_ids", lambda v: _coordinates(v, "product_ids"))):
            object.__setattr__(self, name, check(getattr(self, name)))
        m, b, p = len(self.member_ids), len(self.biome_ids), len(self.product_ids)
        for name, shape in (("cwd_loss", (m, b)), ("soil_loss", (m, b)),
                            ("cwd_resp_fraction", (m, b)), ("hwp_loss", (m, b, p))):
            object.__setattr__(self, name, _carbon_array(getattr(self, name), shape, name))
        if np.any(self.cwd_resp_fraction > 1):
            raise ValueError("CWD respiration fractions must lie in [0,1]")


@dataclass(frozen=True)
class DeadCarbonStep:
    cwd: np.ndarray
    soil: np.ndarray
    hwp: np.ndarray
    cwd_input: np.ndarray
    soil_input: np.ndarray
    hwp_input: np.ndarray
    cwd_to_soil: np.ndarray
    cwd_respiration: np.ndarray
    soil_respiration: np.ndarray
    hwp_respiration: np.ndarray

    @property
    def total_carbon(self):
        return (self.cwd + self.soil).sum(axis=1) + self.hwp.sum(axis=(1, 2))

    @property
    def atmospheric_loss(self):
        return (self.cwd_respiration + self.soil_respiration).sum(axis=1) + self.hwp_respiration.sum(axis=(1, 2))

    @property
    def external_input(self):
        return (self.cwd_input + self.soil_input).sum(axis=1) + self.hwp_input.sum(axis=(1, 2))


def _cascade(cwd, soil, cwd_input, soil_input, a, b, fraction, dt):
    """Positive uniformization of the constant-coefficient two-pool cascade.

    q >= both loss rates makes I+L/q nonnegative, including the cumulative
    transfer rows. Each subinterval has s=q*dt<=1. Terms n=0,...,20 of the
    Poisson series approximate exp(L*dt); constant inputs are an augmented
    constant coordinate. At s=1 the omitted probability and first moment are
    <8e-21 and <2e-19. No small difference of rates or stocks is divided/subtracted.
    Weighted terms use input*dt, not input/q, to handle arbitrarily small q.
    """
    q = max(float(np.max(a)), float(np.max(b)))
    transfers = [np.zeros_like(cwd) for _ in range(3)]
    if dt == 0 or q == 0:
        return cwd + cwd_input * dt, soil + soil_input * dt, *transfers
    if not math.isfinite(q * dt):
        raise ValueError("loss rate times dt is not finite")
    substeps = max(1, math.ceil(q * dt))
    width = dt / substeps
    s, u, v = q * width, a / q, b / q
    for _ in range(substeps):
        weight = math.exp(-s)
        # Order: CWD, soil, CWD->soil, CWD->air, soil->air.
        term = [weight * cwd, weight * soil, *[np.zeros_like(cwd) for _ in range(3)]]
        total = [value.copy() for value in term]
        for n in range(1, 21):
            factor = s / n
            x, y, moved, ex, ey = term
            xloss, yloss = u * x, v * y
            transfer = (1 - fraction) * xloss
            term = [factor * (1-u) * x + cwd_input * (width/n) * weight,
                    factor * ((1-v) * y + transfer) + soil_input * (width/n) * weight,
                    factor * (moved + transfer),
                    factor * (ex + fraction * xloss),
                    factor * (ey + yloss)]
            for j in range(5):
                total[j] += term[j]
            weight *= factor
        cwd, soil = total[:2]
        for j in range(3):
            transfers[j] += total[j+2]
    return cwd, soil, *transfers


def advance_dead_carbon(cwd, soil, hwp, cwd_input_rate, soil_input_rate,
                        hwp_input_rate, rates, dt, *, member_ids, biome_ids, product_ids):
    """Advance actual stocks and every transfer using frozen rates and inputs.

    Member order is explicitly aligned; biome/product axes must already match
    the accepted data interface. External input is a rate, never a second copy
    of an already credited stock transfer. Supplying a live-step integrated
    transfer divided by dt treats that input as uniform within the step: this
    coupling is second-order, not an exact joint live/dead solution.
    """
    ids = _member_ids(member_ids)
    if tuple(biome_ids) != rates.biome_ids or tuple(product_ids) != rates.product_ids:
        raise ValueError("biome/product coordinate order differs from rates")
    shape = (len(ids), len(rates.biome_ids))
    product_shape = shape + (len(rates.product_ids),)
    cwd, soil, cwd_in, soil_in = [
        _carbon_array(value, shape, name) for value, name in
        ((cwd, "CWD"), (soil, "soil"), (cwd_input_rate, "CWD input"), (soil_input_rate, "soil input"))]
    hwp = _carbon_array(hwp, product_shape, "HWP")
    hwp_in = _carbon_array(hwp_input_rate, product_shape, "HWP input")
    dt = _scalar(dt, "dt")
    a, b, fraction, hwp_rate = [align_members(getattr(rates, name), rates.member_ids, ids)
                               for name in ("cwd_loss", "soil_loss", "cwd_resp_fraction", "hwp_loss")]
    x, y, transfer, ex, ey = _cascade(cwd, soil, cwd_in, soil_in, a, b, fraction, dt)
    products = advance_live_carbon(hwp, hwp_in, hwp_rate, 0., dt)
    result = DeadCarbonStep(x, y, products.carbon, cwd_in*dt, soil_in*dt,
                            products.production, transfer, ex, ey, products.mortality)
    for name, value in result.__dict__.items():
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError(f"nonfinite or negative integrated {name}")
    return result

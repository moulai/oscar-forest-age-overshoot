"""Runtime evaluator for the accepted regional forest-age response curves.

The evaluator consumes only the generic Hermite coefficients and matched mature
tail produced by the registered curve workflow. It never reads source rasters,
refits parameters, or multiplies by forest area. Returned stock response S is
dimensionless and its derivative is yr^-1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

CURVE_SCHEMA = "regional-continuous-curves-runtime-v1"


def _labels(values, name):
    values = tuple(str(x) for x in values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    return values


@dataclass(frozen=True)
class GrowthCurve:
    region_ids: tuple
    weighting: str
    knot_age: np.ndarray
    coefficient: np.ndarray
    tail_gap: np.ndarray
    tail_rate: np.ndarray

    def __post_init__(self):
        regions = _labels(self.region_ids, "region_ids")
        if self.weighting not in ("capacity", "area"):
            raise ValueError("weighting must be capacity or area")
        # Own the arrays: freezing a view would also change the caller's flags.
        knots = np.array(self.knot_age, dtype=np.float64, copy=True)
        coeff = np.array(self.coefficient, dtype=np.float64, copy=True)
        gap = np.array(self.tail_gap, dtype=np.float64, copy=True)
        rate = np.array(self.tail_rate, dtype=np.float64, copy=True)
        if knots.ndim != 1 or knots.size < 2 or not np.all(np.isfinite(knots)):
            raise ValueError("knot_age must be a finite one-dimensional array")
        if knots[0] != 0 or not np.all(np.diff(knots) > 0):
            raise ValueError("knot_age must start at zero and increase strictly")
        expected = (len(regions), knots.size - 1, 4)
        if coeff.shape != expected or not np.all(np.isfinite(coeff)):
            raise ValueError("coefficient has the wrong shape or nonfinite values")
        if gap.shape != (len(regions),) or rate.shape != (len(regions),):
            raise ValueError("tail arrays have the wrong shape")
        if not np.all(np.isfinite(gap)) or not np.all(np.isfinite(rate)) or np.any(gap <= 0) or np.any(rate <= 0):
            raise ValueError("tail gap and rate must be finite and positive")
        minimum = np.zeros((len(regions), knots.size - 1))
        c1, c2, c3 = coeff[..., 1], coeff[..., 2], coeff[..., 3]
        h = np.diff(knots)[None, :]
        endpoint = c1 + 2*c2*h + 3*c3*h*h
        vertex = np.divide(-c2, 3*c3, out=np.full_like(c2, np.nan), where=c3 != 0)
        inside = (vertex > 0) & (vertex < h)
        minimum[...] = np.minimum(c1, endpoint)
        minimum[inside] = np.minimum(minimum[inside],
            c1[inside] + 2*c2[inside]*vertex[inside] + 3*c3[inside]*vertex[inside]**2)
        if np.any(minimum < -1e-12):
            raise ValueError("stored Hermite curve is not monotone")
        right_stock = coeff[..., 0] + h*(c1 + h*(c2 + h*c3))
        if (np.any(coeff[..., 0] < -1e-12) or np.any(right_stock > 1 + 1e-12)
                or not np.allclose(coeff[:, 0, 0], 0, rtol=0, atol=1e-12)):
            raise ValueError("stored curve must start at zero and stay within [0,1]")
        if (not np.allclose(right_stock[:, :-1], coeff[:, 1:, 0], rtol=0, atol=1e-12)
                or not np.allclose(endpoint[:, :-1], c1[:, 1:], rtol=0, atol=1e-12)
                or not np.allclose(right_stock[:, -1], 1-gap, rtol=0, atol=1e-12)
                or not np.allclose(endpoint[:, -1], gap*rate, rtol=0, atol=1e-12)):
            raise ValueError("stored curve and mature tail must be C1 continuous")
        knots.setflags(write=False); coeff.setflags(write=False)
        gap.setflags(write=False); rate.setflags(write=False)
        object.__setattr__(self, "region_ids", regions)
        object.__setattr__(self, "knot_age", knots)
        object.__setattr__(self, "coefficient", coeff)
        object.__setattr__(self, "tail_gap", gap)
        object.__setattr__(self, "tail_rate", rate)

    def _indices(self, regions: Iterable[str] | None):
        if regions is None:
            return tuple(range(len(self.region_ids))), self.region_ids
        requested = tuple(str(x) for x in regions)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("requested regions must be nonempty and unique")
        missing = set(requested) - set(self.region_ids)
        if missing:
            raise ValueError("unknown region: " + ",".join(sorted(missing)))
        lookup = {name: i for i, name in enumerate(self.region_ids)}
        return tuple(lookup[name] for name in requested), requested

    def evaluate(self, age, *, regions=None):
        """Return (S, dS/da) with shape (region,)+broadcasted age shape."""
        ages = np.asarray(age, dtype=np.float64)
        if not np.all(np.isfinite(ages)) or np.any(ages < 0):
            raise ValueError("age must be finite and nonnegative")
        indices, _ = self._indices(regions)
        flat = ages.reshape(-1)
        stock = np.empty((len(indices), flat.size), dtype=np.float64)
        slope = np.empty_like(stock)
        last = float(self.knot_age[-1])
        for out, index in enumerate(indices):
            tail = flat >= last
            stock[out, tail] = 1 - self.tail_gap[index] * np.exp(
                -self.tail_rate[index] * (flat[tail] - last))
            slope[out, tail] = self.tail_gap[index] * self.tail_rate[index] * np.exp(
                -self.tail_rate[index] * (flat[tail] - last))
            finite = ~tail
            if np.any(finite):
                local = flat[finite]
                interval = np.searchsorted(self.knot_age, local, side="right") - 1
                x = local - self.knot_age[interval]
                c = self.coefficient[index, interval]
                stock[out, finite] = c[:, 0] + x*(c[:, 1] + x*(c[:, 2] + x*c[:, 3]))
                slope[out, finite] = c[:, 1] + x*(2*c[:, 2] + 3*x*c[:, 3])
        stock = stock.reshape((len(indices),) + ages.shape)
        slope = slope.reshape(stock.shape)
        if np.any(stock < -1e-10) or np.any(stock > 1 + 1e-10) or np.any(slope < -1e-10):
            raise ValueError("evaluated curve violates [0,1] or monotonicity bounds")
        return stock, slope


def load_growth_curve(path, *, weighting="capacity", regions=None):
    """Load an accepted generic curve and optionally reorder its regions."""
    path = Path(path)
    with xr.open_dataset(path) as opened:
        ds = opened.load()
    if weighting not in ("capacity", "area"):
        raise ValueError("weighting must be capacity or area")
    dimensions = {"coefficient": ("weighting", "reg_land", "interval", "power"),
                  "knot_age": ("knot",), "tail_gap": ("weighting", "reg_land"),
                  "tail_rate": ("weighting", "reg_land")}
    required = set(dimensions)
    if not required.issubset(ds.data_vars):
        raise ValueError("curve artifact is missing required variables")
    if "weighting" not in ds.coords or weighting not in ds.weighting.values:
        raise ValueError("requested weighting is unavailable")
    if "reg_land" not in ds.coords:
        raise ValueError("curve artifact is missing reg_land coordinate")
    for name, dims in dimensions.items():
        if set(ds[name].dims) != set(dims):
            raise ValueError(f"curve artifact has malformed {name} dimensions")
    for name in ("reg_land", "weighting", "knot", "interval", "power"):
        if name not in ds.coords or ds[name].dims != (name,):
            raise ValueError(f"curve artifact has malformed {name} coordinate")
    _labels(ds.weighting.values, "weighting")
    for name in ("knot", "interval", "power"):
        if not np.array_equal(ds[name].values, np.arange(ds.sizes[name])):
            raise ValueError(f"curve artifact has malformed {name} ordering")
    source_regions = _labels(ds.reg_land.values, "reg_land")
    requested = source_regions if regions is None else _labels(regions, "requested regions")
    if set(requested) != set(source_regions):
        raise ValueError("requested regions must match the complete artifact axis")
    selected = ds.sel(weighting=weighting, reg_land=list(requested))
    knots = np.asarray(selected.knot_age.values, dtype=np.float64)
    coeff = np.asarray(selected.coefficient.transpose("reg_land", "interval", "power").values)
    gap = np.asarray(selected.tail_gap.transpose("reg_land").values)
    rate = np.asarray(selected.tail_rate.transpose("reg_land").values)
    curve = GrowthCurve(requested, str(weighting), knots, coeff, gap, rate)
    if curve.knot_age[-1] != 500:
        raise ValueError("accepted runtime contract requires the 500-year tail boundary")
    return curve

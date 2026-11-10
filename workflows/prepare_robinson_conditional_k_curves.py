"""Reduce central and conditional k +/- reported-SE curves on fixed forest support.

This is an input preparation step, not a model integration or a joint uncertainty
interval. A, b, m=2/3, the existing forest mask, weights and Hermite knots stay fixed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import rasterio
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "inputs/raw/Robinson2025_v1"
SUPPORT = ROOT / "inputs/robinson_k_error_support/forest850_region_weights.nc"
REFERENCE = ROOT / "data/processed/regional_forest_age_inputs_v1/growth_curve.nc"
OUTPUT = ROOT / "outputs/01_prepared_data/robinson_conditional_k_v1"
VARIANTS = (("central_reduced", 0), ("k_minus_se", -1), ("k_plus_se", 1))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coefficients(knots, stock, slope):
    """Same local-age cubic Hermite formula used for the main curve artifact."""
    h = np.diff(knots)
    secant = np.diff(stock) / h
    return np.stack((stock[:-1], slope[:-1],
                     (3*secant - 2*slope[:-1] - slope[1:]) / h,
                     (slope[:-1] + slope[1:] - 2*secant) / (h*h)), axis=-1)


def minimum_derivative(knots, coefs):
    h = np.diff(knots)
    c1, c2, c3 = coefs[:, 1], coefs[:, 2], coefs[:, 3]
    minimum = np.minimum(c1, c1 + 2*c2*h + 3*c3*h*h)
    vertex = np.divide(-c2, 3*c3, out=np.full_like(c2, np.nan), where=c3 != 0)
    inside = (vertex > 0) & (vertex < h)
    minimum[inside] = np.minimum(minimum[inside],
        c1[inside] + 2*c2[inside]*vertex[inside] + 3*c3[inside]*vertex[inside]**2)
    return minimum


def build_curve(reference, static, nodes, knots, sign):
    """Keep both original weightings and the original 500-year matched tail."""
    nr, na, _ = nodes.shape
    valid_area, sum_a, sum_b0, sum_capacity = static[:, 1:].T
    stock = np.stack(((nodes[:, :, 0] - sum_b0[:, None])/sum_capacity[:, None],
                      nodes[:, :, 2]/valid_area[:, None]))
    slope = np.stack((nodes[:, :, 1]/sum_capacity[:, None],
                      nodes[:, :, 3]/valid_area[:, None]))
    coefs = np.empty((2, nr, na-1, 4))
    minima = np.empty((2, nr, na-1))
    for w in range(2):
        for r in range(nr):
            coefs[w, r] = coefficients(knots, stock[w, r], slope[w, r])
            minima[w, r] = minimum_derivative(knots, coefs[w, r])
    gap = 1 - stock[:, :, -1]
    rate = slope[:, :, -1]/gap
    require(np.all(minima >= -1e-12) and np.all(gap > 0) and np.all(rate > 0),
            "Conditional input cannot use the existing monotone Hermite/tail representation")
    result = reference.copy(deep=True)
    result["geometric_forest_area"].values[:] = static[:, 0]
    result["valid_forest_area"].values[:] = valid_area
    result["missing_forest_area"].values[:] = result.source_forest_area.values - valid_area
    result["coverage"].values[:] = valid_area/result.source_forest_area.values
    result["A_mean"].values[:] = sum_a/valid_area
    result["B0_mean"].values[:] = sum_b0/valid_area
    result["capacity_mean"].values[:] = sum_capacity/valid_area
    result["coefficient"].values[:] = coefs
    result["minimum_interval_derivative"].values[:] = minima
    result["tail_gap"].values[:] = gap
    result["tail_rate"].values[:] = rate
    result.attrs.update(conditional_k_se_sign=sign, m="2/3", q=3,
        scope="Conditional marginal k perturbation at central A and b; not a joint confidence interval or whole-NPP uncertainty.",
        intended_application="Post-2022-born wood production response only; existing cohorts, alpha, lambda calibration and foliage response remain unchanged.",
        standard_error_source="Zenodo 15090826 version 1 cr_k_error.tif; marginal nonlinear-least-squares standard error.",
        reduction="Unchanged 850 forest mask and spherical/native-region weights; normalize regional B-B0 by regional A-B0; same knots, Hermite formula and matched tail.",
        domain_handling="Same original centre mask in every variant; no clipping, filling, replacement or reweighting.")
    return result, stock, slope


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(schema="robinson-conditional-k-curves-v1", status="running", started_utc=now(),
        source_manifest="inputs/raw/Robinson2025_v1/source_manifest.json",
        coverage_summary="outputs/01_prepared_data/robinson_k_error_support_v1/summary.json",
        fixed_support=str(SUPPORT.relative_to(ROOT)), reference_curve=str(REFERENCE.relative_to(ROOT)),
        reference_curve_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        m="2/3", q=3, central_A_b_fixed=True, same_mask_and_capacity_weights=True,
        clipping_or_domain_completion=False, model_integrations=0)
    save(output/"manifest.json", manifest)
    try:
        coverage = json.loads((ROOT/manifest["coverage_summary"]).read_text())
        require(coverage["status"] == "complete", "Complete input coverage is required")
        with xr.open_dataset(SUPPORT) as opened:
            weights = opened.load()
        with xr.open_dataset(REFERENCE) as opened:
            reference = opened.load()
        require(np.array_equal(weights.reg_land.values, reference.reg_land.values), "Region order differs")
        require(list(reference.weighting.values) == ["capacity", "area"], "Unexpected weighting order")
        knots = reference.knot_age.values
        pi = weights.region_fraction.values.reshape(-1, weights.sizes["reg_land"])
        forest, row_fraction = weights.forest850_area.values, weights.fine_row_area_fraction.values
        nr, na = reference.sizes["reg_land"], len(knots)
        static = np.zeros((nr, 5))
        node_sums = np.zeros((3, nr, na, 4))
        with ExitStack() as stack:
            readers = {key:stack.enter_context(rasterio.open(RAW/name)) for key, name in
                (("A", "cr_a.tif"), ("b", "cr_b.tif"), ("k", "cr_k.tif"), ("se", "cr_k_error.tif"))}
            ref = readers["A"]
            require(ref.shape == (21120, 43200) and ref.crs.to_epsg() == 4326, "Unexpected raw grid")
            for reader in readers.values():
                require((reader.shape, reader.transform, reader.crs) == (ref.shape, ref.transform, ref.crs), "Raster grids differ")
            for index, (_, window) in enumerate(ref.block_windows(1)):
                rows = np.arange(int(window.row_off), int(window.row_off+window.height))
                cols = np.arange(int(window.col_off), int(window.col_off+window.width))
                qr, qc = 8+rows//30, cols//30
                area = forest[qr[:, None], qc[None, :]] * row_fraction[rows, None]/30.
                if not np.any(area > 0):
                    continue
                region_cells = (qr[:, None]//4)*360 + qc[None, :]//4
                cells, reverse = np.unique(region_cells, return_inverse=True)
                static[:, 0] += pi[cells].T @ np.bincount(reverse.ravel(), weights=area.ravel(), minlength=len(cells))
                masked = {key:reader.read(1, window=window, masked=True) for key, reader in readers.items()}
                A, b, k, se = [np.asarray(masked[key].data, dtype=np.float64) for key in ("A", "b", "k", "se")]
                valid = (area > 0) & np.isfinite(A) & np.isfinite(b) & np.isfinite(k)
                valid &= (A > 0) & (b > 0) & (b <= 1) & (k > 0)
                for key in ("A", "b", "k"):
                    valid &= ~np.ma.getmaskarray(masked[key])
                if not np.any(valid):
                    continue
                require(np.all(np.isfinite(se[valid]) & (se[valid] >= 0) & (k[valid]-se[valid] > 0))
                        and not np.any(np.ma.getmaskarray(masked["se"])[valid]), "SE would change the original valid domain")
                aa, bb, kk, ee, ww = [v[valid] for v in (A, b, k, se, area)]
                cells, reverse = np.unique(region_cells[valid], return_inverse=True)
                region_weights = pi[cells].T
                b0 = aa*(1-bb)**3
                capacity = aa-b0
                stats = np.stack([np.bincount(reverse, weights=ww*v, minlength=len(cells))
                                  for v in (np.ones_like(aa), aa, b0, capacity)], axis=1)
                static[:, 1:] += region_weights @ stats
                for variant, (_, sign) in enumerate(VARIANTS):
                    rate = kk + sign*ee
                    for j, age in enumerate(knots):
                        x = bb*np.exp(-rate*age)
                        stock = aa*(1-x)**3
                        slope = 3*aa*rate*x*(1-x)**2
                        stats = np.stack([np.bincount(reverse, weights=ww*v, minlength=len(cells))
                            for v in (stock, slope, (stock-b0)/capacity, slope/capacity)], axis=1)
                        node_sums[variant, :, j] += region_weights @ stats
                if index % 500 == 0:
                    manifest.update(blocks_seen=index+1, elapsed_seconds=time.monotonic()-started)
                    save(output/"manifest.json", manifest)
                    print(json.dumps(dict(event="curve_reduction_progress", blocks_seen=index+1,
                        elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
        require(np.all(static[:, 1] > 0) and np.all(static[:, 4] > 0), "A native region lacks the fixed support")
        artifacts = {}
        rows = []
        for variant, (name, sign) in enumerate(VARIANTS):
            curve, stock, slope = build_curve(reference, static, node_sums[variant], knots, sign)
            if sign == 0:
                errors = {key:float(np.max(np.abs(curve[key].values-reference[key].values))) for key in reference.data_vars}
                for key in reference.data_vars:
                    require(np.allclose(curve[key].values, reference[key].values, rtol=1e-12, atol=1e-12),
                            "Central reduction differs beyond rounding: " + key)
                manifest["central_input_isolation_max_abs"] = errors
            path = output/("growth_curve_" + name + ".nc")
            curve.to_netcdf(path)
            from oscar._core._model._modules.growth_curve import load_growth_curve
            for weighting in ("capacity", "area"):
                load_growth_curve(path, weighting=weighting)
            artifacts[name] = dict(path=str(path.relative_to(ROOT)), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), k_standard_error_sign=sign,
                weightings=["capacity", "area"], native_region_count=nr, knot_count=na)
            for w, weighting in enumerate(("capacity", "area")):
                for r, region in enumerate(reference.reg_land.values):
                    for j, age in enumerate(knots):
                        rows.append((name, weighting, str(region), float(age), stock[w, r, j], slope[w, r, j]))
        import csv
        with (output/"regional_curve_nodes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("condition", "weighting", "reg_land", "age_years", "S", "S_prime_per_year"))
            writer.writerows(rows)
        manifest.update(status="ready", completed_utc=now(), elapsed_seconds=time.monotonic()-started,
            curves=artifacts, raster_passes=1,
            experimental_scope="Conditional k +/- one reported NLS standard error at fixed A,b; future-born W response only. Not a joint curve CI or foliage-establishment bound.",
            unchanged_model_inputs=["Original main growth curve", "alpha", "lambda_w_cal", "F establishment timescale", "2022 initial state and existing-cohort responses"])
        save(output/"manifest.json", manifest)
        print(json.dumps(dict(status="ready", output=str(output.relative_to(ROOT)), curves=artifacts)), flush=True)
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc), updated_utc=now(), elapsed_seconds=time.monotonic()-started)
        save(output/"manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)


if __name__ == "__main__":
    main()

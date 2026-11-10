"""Acquire Robinson v1 k standard errors and describe their existing forest support.

This input-only workflow does not generate a perturbed curve or integrate OSCAR.
The coverage reduction uses the unchanged 850 forest/region weights and q=3.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_NUM_THREADS"):
    os.environ[_key] = "1"

ROOT = Path(__file__).resolve().parents[1]
API = "https://zenodo.org/api/records/15090826"
RAW = ROOT / "inputs/raw/Robinson2025_v1"
SUPPORT = ROOT / "inputs/robinson_k_error_support"
OUTPUT = ROOT / "outputs/01_prepared_data/robinson_k_error_support_v1"
NAMES = ("cr_a.tif", "cr_b.tif", "cr_k.tif", "cr_k_error.tif", "ReadMe.md")


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def remote_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "OSCAR-Forest-Age scientific-input preparation"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def download_k_error(item, partial):
    """Finish this one official file with four disjoint, checked HTTP ranges."""
    marker = partial.with_name(partial.name + ".ranges.json")
    if marker.exists():
        raise RuntimeError("Inspect the retained incomplete range receipt before another acquisition")
    prefix = partial.stat().st_size if partial.exists() else 0
    size = item["size"]
    if not 0 <= prefix < size:
        raise RuntimeError("Unexpected sequential-download prefix size")
    ranges = [(prefix+(size-prefix)*i//4, prefix+(size-prefix)*(i+1)//4-1) for i in range(4)]
    write(marker, dict(status="running", prefix_bytes=prefix, ranges=ranges, started_utc=now()))
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    def get_range(bounds):
        left, right = bounds
        request = urllib.request.Request(item["links"]["self"], headers={
            "User-Agent":"OSCAR-Forest-Age scientific-input preparation", "Range":f"bytes={left}-{right}"})
        position = left
        with urllib.request.urlopen(request, timeout=180) as incoming:
            if incoming.status != 206 or incoming.headers.get("Content-Range") != f"bytes {left}-{right}/{size}":
                raise RuntimeError("Provider did not honour the disjoint byte range")
            while True:
                block = incoming.read(1024*1024)
                if not block:
                    break
                if position+len(block)>right+1:
                    raise RuntimeError("Provider response exceeds requested range")
                os.pwrite(fd, block, position)
                position += len(block)
        if position != right+1:
            raise RuntimeError("Incomplete official byte range")
        return dict(left=left, right=right, bytes=position-left)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(get_range, bounds) for bounds in ranges]):
                print(json.dumps(dict(event="official_range_complete", **future.result(), utc=now())), flush=True)
    finally:
        os.close(fd)
    write(marker, dict(status="downloaded_pending_checksum", prefix_bytes=prefix, ranges=ranges, completed_utc=now()))


def acquire(args):
    RAW.mkdir(parents=True, exist_ok=True)
    SUPPORT.mkdir(parents=True, exist_ok=True)
    record = remote_json(API)
    files = {entry["key"]: entry for entry in record["files"]}
    manifest_path = RAW / "source_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = dict(schema="robinson-k-error-input-v1", status="acquiring", started_utc=now(),
        record_url="https://zenodo.org/records/15090826", api_url=API, doi=record["doi"],
        version=record["metadata"].get("version"), publication_date=record["metadata"].get("publication_date"),
        licence=record["metadata"].get("license"),
        error_definition="Marginal NLS parameter standard error, not joint curve uncertainty or an NPP uncertainty interval.",
        shape_convention=dict(m="2/3", q=3,
            source="Existing study convention, matching the paper and fixed author v1.0 fitting script; unchanged here."),
        files=previous.get("files", {}))
    write(manifest_path, manifest)
    for name in NAMES:
        item, target = files[name], RAW / name
        expected = item["checksum"].removeprefix("md5:")
        old = manifest["files"].get(name, {})
        if target.exists():
            if target.stat().st_size != item["size"] or old.get("verified_md5") != expected:
                raise RuntimeError("Existing input is not a completed acquisition: " + str(target))
            continue
        partial = target.with_name(target.name + ".partial")
        if partial.exists() and name!="cr_k_error.tif":
            raise FileExistsError("An incomplete acquisition is retained; inspect before restarting: " + str(partial))
        source = args.reuse_centres_from / name if name in NAMES[:3] else None
        digest, received = hashlib.md5(), 0
        request = urllib.request.Request(item["links"]["self"], headers={"User-Agent": "OSCAR-Forest-Age scientific-input preparation"})
        if name=="cr_k_error.tif":
            download_k_error(item, partial)
            with partial.open("rb") as incoming:
                for block in iter(lambda:incoming.read(8*1024*1024), b""):
                    digest.update(block);received+=len(block)
        else:
            with ExitStack() as stack:
                incoming = stack.enter_context(source.open("rb") if source is not None else urllib.request.urlopen(request, timeout=180))
                outgoing = stack.enter_context(partial.open("xb"))
                while True:
                    block = incoming.read(8 * 1024 * 1024)
                    if not block:
                        break
                    outgoing.write(block)
                    digest.update(block)
                    received += len(block)
        if received != item["size"] or digest.hexdigest() != expected:
            raise RuntimeError("Provider bytes/checksum differ for " + name)
        partial.replace(target)
        if name=="cr_k_error.tif":
            partial.with_name(partial.name+".ranges.json").unlink()
        manifest["files"][name] = dict(path=str(target.relative_to(ROOT)), provider_url=item["links"]["self"],
            size_bytes=received, provider_checksum=item["checksum"], verified_md5=digest.hexdigest(),
            acquisition="directed copy of the previously acquired official file" if source is not None else "official download",
            previous_project_source=str(source) if source is not None else None, acquired_utc=now())
        write(manifest_path, manifest)
        print(json.dumps(dict(event="input_acquired", name=name, bytes=received, utc=now())), flush=True)
    weight_path = SUPPORT / "forest850_region_weights.nc"
    if not weight_path.exists():
        shutil.copy2(args.reuse_support_from, weight_path)
    manifest["forest_support"] = dict(path=str(weight_path.relative_to(ROOT)),
        copied_from=str(args.reuse_support_from), size_bytes=weight_path.stat().st_size,
        interpretation="Unchanged original LUH850 forest areas, spherical subdivisions and native regional fractions; runtime reduction reads only this project-local copy.")
    manifest.update(status="complete", completed_utc=now())
    write(manifest_path, manifest)
    write(SUPPORT / "source_manifest.json", {k:manifest[k] for k in ("schema", "forest_support", "shape_convention", "doi", "version")})


def coverage(args):
    import numpy as np
    import rasterio
    import xarray as xr
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW / "source_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Raw acquisition is incomplete")
    with xr.open_dataset(SUPPORT / "forest850_region_weights.nc") as opened:
        weights = opened.load()
    pi = weights.region_fraction.values.reshape(-1, weights.sizes["reg_land"])
    forest, row_fraction = weights.forest850_area.values, weights.fine_row_area_fraction.values
    regions = list(map(str, weights.reg_land.values))
    keys = ("geometric_area", "central_valid_area", "central_capacity", "se_valid_area", "se_valid_capacity",
            "se_missing_area", "se_missing_capacity", "se_negative_area", "se_negative_capacity",
            "k_minus_nonpositive_area", "k_minus_nonpositive_capacity", "k_plus_invalid_area", "k_plus_invalid_capacity",
            "se_zero_area", "se_zero_capacity", "se_ratio_ge10_area", "se_ratio_ge10_capacity",
            "se_ratio_ge100_area", "se_ratio_ge100_capacity", "se_ratio_ge1e6_area", "se_ratio_ge1e6_capacity")
    totals = np.zeros((len(regions), len(keys)), dtype=float)
    # Streaming input summaries: retain quantile-bin bounds instead of claiming
    # exact pixel quantiles or sorting hundreds of millions of raw values.
    ratio_edges = np.r_[0., np.logspace(-12, 60, 32769), np.inf]
    ratio_hist = np.zeros((2, len(ratio_edges)-1))
    ratio_extremes = [float("inf"), 0.]
    started = time.monotonic()
    report = dict(schema="robinson-k-standard-error-support-v1", status="running", started_utc=now(),
        raw_manifest=str((RAW/"source_manifest.json").relative_to(ROOT)),
        support_manifest=str((SUPPORT/"source_manifest.json").relative_to(ROOT)),
        m="2/3", q=3, clipping_or_domain_completion=False, new_curves_generated=False,
        original_curve="data/processed/regional_forest_age_inputs_v1/growth_curve.nc")
    write(OUTPUT / "status.json", report)
    try:
        with ExitStack() as stack:
            readers = {key:stack.enter_context(rasterio.open(RAW/name)) for key,name in
                       (("A","cr_a.tif"),("b","cr_b.tif"),("k","cr_k.tif"),("se","cr_k_error.tif"))}
            ref = readers["A"]
            if ref.shape != (21120,43200) or ref.crs.to_epsg() != 4326:
                raise RuntimeError("Unexpected original CR grid")
            if not np.allclose(list(ref.transform)[:6], [1/120,0,-180,0,-1/120,88], rtol=0, atol=1e-10):
                raise RuntimeError("Original CR grid is not aligned to the retained forest support")
            for key, reader in readers.items():
                if (reader.shape,reader.transform,reader.crs) != (ref.shape,ref.transform,ref.crs):
                    raise RuntimeError("Different grid: " + key)
                if reader.scales != (1.,) or reader.offsets != (0.,):
                    raise RuntimeError("Unspecified stored-value transform: " + key)
            report["raster_grid"] = dict(shape=list(ref.shape), transform=list(ref.transform)[:6], crs=str(ref.crs))
            for index, (_, window) in enumerate(ref.block_windows(1)):
                rows = np.arange(int(window.row_off), int(window.row_off+window.height))
                cols = np.arange(int(window.col_off), int(window.col_off+window.width))
                qr, qc = 8+rows//30, cols//30
                area = forest[qr[:,None],qc[None,:]] * row_fraction[rows,None] / 30.
                if not np.any(area > 0):
                    continue
                region_cells = (qr[:,None]//4)*360+qc[None,:]//4
                cells, inverse = np.unique(region_cells, return_inverse=True)
                inverse = inverse.reshape(area.shape)
                arrays, masks = {}, {}
                for key, reader in readers.items():
                    x = reader.read(1, window=window, masked=True)
                    arrays[key] = np.asarray(x.data, dtype=float)
                    masks[key] = np.ma.getmaskarray(x)
                A,b,k,se = (arrays[key] for key in ("A","b","k","se"))
                central = (area>0) & np.isfinite(A) & np.isfinite(b) & np.isfinite(k)
                central &= ~(masks["A"] | masks["b"] | masks["k"]) & (A>0) & (b>0) & (b<=1) & (k>0)
                capacity = np.zeros_like(A)
                capacity[central] = A[central] * (1-(1-b[central])**3)
                missing = central & (masks["se"] | ~np.isfinite(se))
                negative = central & ~missing & (se<0)
                valid_se = central & ~missing & ~negative
                minus = valid_se & (k-se<=0)
                plus = valid_se & (~np.isfinite(k+se) | (k+se<=0))
                ratio = np.zeros_like(se)
                np.divide(se, k, out=ratio, where=valid_se)
                weighted = [area, area*central, area*capacity]
                for mask in (valid_se,missing,negative,minus,plus,valid_se&(se==0),
                             valid_se&(ratio>=10),valid_se&(ratio>=100),valid_se&(ratio>=1e6)):
                    weighted += [area*mask, area*capacity*mask]
                block = np.stack([np.bincount(inverse.ravel(),weights=x.ravel(),minlength=len(cells)) for x in weighted],axis=1)
                totals += pi[cells].T @ block
                if np.any(valid_se):
                    values = ratio[valid_se]
                    indices = np.searchsorted(ratio_edges, values, side="right")-1
                    area_weights = area[valid_se] * pi[cells].sum(axis=1)[inverse[valid_se]]
                    for wi, w in enumerate((area_weights, area_weights*capacity[valid_se])):
                        ratio_hist[wi] += np.bincount(indices, weights=w, minlength=len(ratio_edges)-1)
                    ratio_extremes[0] = min(ratio_extremes[0], float(values.min()))
                    ratio_extremes[1] = max(ratio_extremes[1], float(values.max()))
                if index % 250 == 0:
                    report.update(blocks_seen=index+1, elapsed_seconds=time.monotonic()-started)
                    write(OUTPUT/"status.json", report)
                    print(json.dumps(dict(event="coverage_progress", blocks_seen=index+1,
                        elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
        all_rows = []
        for region, values in list(zip(regions, totals)) + [("GLOBAL",totals.sum(axis=0))]:
            raw = dict(zip(keys,map(float,values)))
            row = dict(region=region, source_forest_Mha=float(weights.source_forest_area.sum() if region=="GLOBAL" else weights.source_forest_area.sel(reg_land=region))/10000)
            row.update({key+"_Mha":value/10000 for key,value in raw.items() if key.endswith("area")})
            row.update({key+"_PgC":value*1e-7 for key,value in raw.items() if key.endswith("capacity")})
            for key in ("se_valid","se_missing","se_negative","k_minus_nonpositive","k_plus_invalid", "se_zero",
                        "se_ratio_ge10","se_ratio_ge100","se_ratio_ge1e6"):
                row[key+"_fraction_of_central_area"] = raw[key+"_area"]/raw["central_valid_area"] if raw["central_valid_area"] else None
                row[key+"_fraction_of_central_capacity"] = raw[key+"_capacity"]/raw["central_capacity"] if raw["central_capacity"] else None
            row["k_minus_nonpositive_fraction_of_se_valid_area"] = raw["k_minus_nonpositive_area"]/raw["se_valid_area"] if raw["se_valid_area"] else None
            row["k_minus_nonpositive_fraction_of_se_valid_capacity"] = raw["k_minus_nonpositive_capacity"]/raw["se_valid_capacity"] if raw["se_valid_capacity"] else None
            all_rows.append(row)
        relative_se = dict(definition="SE(k)/k on finite nonnegative SE and original valid forest/capacity support.",
            quantile_method="Single-pass logarithmic histogram. Each reported interval brackets the weighted pixel quantile; estimate is its geometric midpoint. It is not a confidence interval.",
            ratio_min=ratio_extremes[0] if np.isfinite(ratio_extremes[0]) else None,ratio_max=ratio_extremes[1],
            quantile_bin_max_relative_width=float(10**(72/32768)-1))
        for wi, name in enumerate(("forest_area_weighted", "recoverable_capacity_weighted")):
            cumulative=np.cumsum(ratio_hist[wi]);out={}
            for q,label in ((.5,"median"),(.95,"q95")):
                if cumulative[-1]==0:
                    out[label]=None;continue
                index=int(np.searchsorted(cumulative,q*cumulative[-1]))
                low,high=map(float,ratio_edges[index:index+2])
                out[label]=dict(lower=low,upper=high if np.isfinite(high) else None,
                    estimate=float(np.sqrt(low*high)) if low>0 and np.isfinite(high) else None)
            relative_se[name]=out
        with (OUTPUT/"regional_coverage.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        report.update(status="complete",completed_utc=now(),elapsed_seconds=time.monotonic()-started,
            definitions=dict(capacity="Original A-B(0)=A[1-(1-b)^3], not model NPP or a fitted model stock; PgC sums use km2*100ha*MgC/ha/1e9.",
                central_support="Positive original 850 forest area with finite, unmasked A>0, 0<b<=1 and k>0.",
                se_support="Central support with finite, unmasked and nonnegative k standard error.",
                invalid_slow="k-SE<=0 on valid SE support. Retained as invalid; no clipping or replacement.",
                area_weighting="Original spherical fine-row subdivision and original native-region fractions.",
                uncertainty="Marginal NLS SE, not a joint confidence interval; no climate/model integration."),
            relative_standard_error=relative_se,global_summary=all_rows[-1],regional_summary=all_rows[:-1])
        write(OUTPUT/"summary.json", report)
        write(OUTPUT/"status.json", report)
        print(json.dumps(dict(status="complete",output=str(OUTPUT.relative_to(ROOT)),global_summary=all_rows[-1])),flush=True)
    except Exception as exc:
        report.update(status="failed",error=repr(exc),updated_utc=now())
        write(OUTPUT/"status.json", report)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("acquire","coverage"),required=True)
    parser.add_argument("--reuse-centres-from",type=Path)
    parser.add_argument("--reuse-support-from",type=Path)
    args=parser.parse_args()
    if args.phase=="acquire":
        if args.reuse_centres_from is None or args.reuse_support_from is None:
            parser.error("Acquisition requires explicit source paths for the three centres and fixed support.")
        acquire(args)
    else:
        coverage(args)


if __name__ == "__main__":
    main()

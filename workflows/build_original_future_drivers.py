"""Build original-OSCAR future drivers for five CMIP7 scenarios.

Land activity is aggregated from the official LUH3 transitions already reduced
to the IAMC R10 regions. The age-module event_area variable is not read.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "inputs/driver_build_sources"
OUT_DIR = ROOT / "outputs/01_prepared_data/cmip7_n2o_mass_corrected_v1"
HISTORICAL = OUT_DIR / "historical_concentration_forcing.nc"
ARCHIVE = SOURCE_ROOT / "data/raw/future/cmip7_preflight_v1/oscar_v4_beta2_1/forcing_hist.nc"

BIOME = {
    "primf": "Forest", "secdf": "Forest", "pltns": "Forest",
    "primn": "Non-Forest", "secdn": "Non-Forest",
    "c3ann": "Cropland", "c4ann": "Cropland", "c3per": "Cropland",
    "c4per": "Cropland", "c3nfx": "Cropland",
    "pastr": "Pasture", "range": "Pasture",
    "urban": "Urban",
}
ORDER = ["Forest", "Non-Forest", "Cropland", "Pasture", "Urban"]
PRIMARY = {"primf", "primn"}
# Shifting cultivation is the paired secondary exchange of natural land and cropland.
SHIFT_PAIRS = [(0, 2), (1, 2)]
LAND_NAMES = ("D_dA_lcc1", "D_dA_lcc2", "D_dA_shift", "D_dA_wharv1", "D_dA_wharv2")

SCENARIOS = {
    "h": {
        "code": "H",
        "land": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_v3/h/land_use_absolute.nc",
        "report": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_v3/h/report.json",
        "atmosphere": OUT_DIR / "future_h_forcing.nc",
    },
    "hl": {
        "code": "HL",
        "land": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_completion_v1/hl/land_use_absolute.nc",
        "report": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_completion_v1/hl/report.json",
        "atmosphere": OUT_DIR / "future_hl_forcing.nc",
    },
    "m": {
        "code": "M",
        "land": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_v3/m/land_use_absolute.nc",
        "report": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_v3/m/report.json",
        "atmosphere": OUT_DIR / "future_m_forcing.nc",
    },
    "vl": {
        "code": "VL",
        "land": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_v3/vl/land_use_absolute.nc",
        "report": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_v3/vl/report.json",
        "atmosphere": OUT_DIR / "future_vl_forcing.nc",
    },
    "ln": {
        "code": "LN",
        "land": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_completion_v1/ln/land_use_absolute.nc",
        "report": SOURCE_ROOT / "data/processed/land_use/future_luh3_r10_completion_v1/ln/report.json",
        "atmosphere": OUT_DIR / "future_ln_forcing.nc",
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_paths(report_path):
    report = json.loads(report_path.read_text())
    receipts = report["source_receipts"]
    paths = [item["path"] for item in receipts]
    transitions = [path for path in paths if "multiple-transitions" in path]
    states = [path for path in paths if "multiple-states" in path]
    require(len(transitions) == 1 and len(states) == 1, "Expected one states and one transitions receipt")
    raw_receipt = next(item for item in receipts if item["path"] == transitions[0])
    require(int(raw_receipt["bytes"]) > 0, "Official transition size is missing from its preserved source receipt")
    return {
        "states": states[0],
        "transitions": transitions[0],
        "transitions_bytes": int(raw_receipt["bytes"]),
        "report_status": report["status"],
        "scenario_in_report": report.get("scenario"),
    }


def aggregate_land(land_path, years, regions):
    land = xr.open_dataset(land_path)
    require(list(map(str, land.reg_land.values)) == regions, "Region order differs from the historical forcing")
    require(np.array_equal(land.interval.values, np.asarray(years)), "LUH intervals are not 2022-2099")
    require(land.transition_area.attrs.get("units") == "Mha", "Transition area is not in Mha")
    require(land.harvest_area.attrs.get("units") == "Mha", "Harvest area is not in Mha")
    classes_from = list(map(str, land.transition_from.values))
    classes_to = list(map(str, land.transition_to.values))
    harvests = list(map(str, land.harvest_source.values))
    known = set(BIOME) | {"secmf", "secyf", "secnf"}
    unknown = sorted({name for name in classes_from + classes_to + harvests if name not in known})
    require(not unknown, "Unexpected LUH label: " + ", ".join(unknown))
    transitions = np.asarray(land.transition_area.transpose("interval", "reg_land", "transition").values, dtype=np.float64)
    harvest = np.asarray(land.harvest_area.transpose("interval", "reg_land", "harvest_source").values, dtype=np.float64)
    require(np.isfinite(transitions).all() and np.isfinite(harvest).all(), "Non-finite regional LUH area")
    n_year, n_region, _ = transitions.shape
    primary = np.zeros((n_year, n_region, 5, 5))
    secondary = np.zeros((n_year, n_region, 5, 5))
    for index, (source, target) in enumerate(zip(classes_from, classes_to)):
        left, right = ORDER.index(BIOME[source]), ORDER.index(BIOME[target])
        destination = primary if source in PRIMARY else secondary
        destination[:, :, left, right] += transitions[:, :, index]
    wood1 = np.zeros((n_year, n_region, 5))
    wood2 = np.zeros((n_year, n_region, 5))
    for index, name in enumerate(harvests):
        values = harvest[:, :, index]
        if name == "primf":
            wood1[:, :, 0] += values
        elif name == "primn":
            wood1[:, :, 1] += values
        elif name in ("secmf", "secyf", "pltns"):
            wood2[:, :, 0] += values
        elif name == "secnf":
            wood2[:, :, 1] += values
        else:
            raise RuntimeError("Unexpected harvest source: " + name)
    shift = np.zeros_like(secondary)
    cover2 = secondary.copy()
    for left, right in SHIFT_PAIRS:
        paired = np.minimum(secondary[:, :, left, right], secondary[:, :, right, left])
        shift[:, :, left, right] = paired
        shift[:, :, right, left] = paired
        cover2[:, :, left, right] -= paired
        cover2[:, :, right, left] -= paired
    land.close()
    return {"D_dA_lcc1": primary, "D_dA_lcc2": cover2, "D_dA_shift": shift,
            "D_dA_wharv1": wood1, "D_dA_wharv2": wood2}


def as_data(name, values, years, regions, historical):
    if name in ("D_dA_wharv1", "D_dA_wharv2"):
        dims = ("year", "reg_land", "bio_land")
        coords = {"year": years, "reg_land": regions, "bio_land": ORDER}
    else:
        dims = ("year", "reg_land", "bio_from", "bio_to")
        coords = {"year": years, "reg_land": regions, "bio_from": ORDER, "bio_to": ORDER}
    array = xr.DataArray(values, dims=dims, coords=coords)
    array.attrs["units"] = "Mha yr-1"
    array.attrs["source"] = historical[name].attrs.get("source", "LUH3 transitions aggregated in this study")
    return array


def continuity(built, historical):
    report = {}
    largest = 0.0
    for name in LAND_NAMES:
        difference = np.abs(built[name].sel(year=2022).values - historical[name].sel(year=2022).values)
        maximum = float(np.max(difference))
        report[name] = {"max_abs_Mha_yr-1": maximum, "sum_built": float(built[name].sel(year=2022).sum()),
                        "sum_historical": float(historical[name].sel(year=2022).sum())}
        largest = max(largest, maximum)
    diagonal = np.abs(built["D_dA_lcc2"].sel(year=2022, bio_from="Cropland", bio_to="Cropland").values
                      - historical["D_dA_lcc2"].sel(year=2022, bio_from="Cropland", bio_to="Cropland").values)
    report["cropland_diagonal_lcc2_max_abs_Mha_yr-1"] = float(np.max(diagonal))
    report["all_fields_max_abs_Mha_yr-1"] = largest
    return report


def build_one(key, spec, historical, years, regions, pi_shift, pi_wood):
    sources = source_paths(spec["report"])
    raw = aggregate_land(spec["land"], years, regions)
    raw["D_dA_shift"] = raw["D_dA_shift"] - pi_shift
    raw["D_dA_wharv2"] = raw["D_dA_wharv2"] - pi_wood
    built = xr.Dataset({name: as_data(name, raw[name], years, regions, historical) for name in LAND_NAMES})
    built.attrs["land_definition"] = (
        "Absolute primary cover change and primary wood harvest; "
        "secondary wood harvest and shifting cultivation minus the historical 1750 baseline"
    )
    check = continuity(built, historical)
    atmosphere = xr.load_dataset(spec["atmosphere"])
    require(atmosphere.attrs.get("n2o_mass_basis") == "TgN_two_nitrogen_atoms_v1", "Atmosphere is not N2O-corrected")
    require("event_area" in atmosphere, "Expected the age-module atmosphere file, which still carries event_area")
    atmosphere = atmosphere.drop_vars("event_area")
    if "event" in atmosphere.coords:
        atmosphere = atmosphere.drop_vars("event")
    require(np.array_equal(atmosphere.year.values, np.asarray(years)), "Atmospheric years are not 2022-2099")
    require(list(map(str, atmosphere.reg_land.values)) == regions, "Atmospheric region order differs")
    static_names = [name for name, value in historical.data_vars.items() if "year" not in value.dims]
    static = historical[static_names]
    merged = xr.merge([atmosphere, built, static], join="exact", compat="no_conflicts")
    require("event_area" not in merged, "event_area leaked into the original forcing")
    for name in LAND_NAMES:
        require(name in merged, "Missing " + name)
    merged.attrs = {
        "role": "Original OSCAR CMIP7 future forcing, 2022 anchor through 2099",
        "n2o_mass_basis": "TgN_two_nitrogen_atoms_v1",
        "n2o_source": str(spec["atmosphere"].relative_to(ROOT)),
        "land_source": "Official LUH3 transitions aggregated in this study; not event_area",
        "time_contract": "2022 is the restart anchor. Integration applies 2023-2099. Year Y is LUH interval Y.",
        "scenario": spec["code"],
    }
    destination = OUT_DIR / f"future_{key}_original_forcing.nc"
    encoding = {name: {"dtype": "float64", "zlib": True, "complevel": 1}
                for name, value in merged.data_vars.items() if value.dtype.kind == "f"}
    merged.to_netcdf(destination, engine="netcdf4", encoding=encoding)
    with xr.open_dataset(destination) as written:
        xr.testing.assert_allclose(written.load(), merged)
    return {
        "scenario": spec["code"],
        "forcing": str(destination.relative_to(ROOT)),
        "regional_land_product": str(spec["land"].relative_to(SOURCE_ROOT)),
        "official_luh": sources,
        "atmosphere": str(spec["atmosphere"].relative_to(ROOT)),
        "continuity_2022_against_historical_forcing": check,
    }


def main():
    historical = xr.load_dataset(HISTORICAL)
    require(historical.attrs.get("n2o_mass_basis") == "TgN_two_nitrogen_atoms_v1", "Historical forcing N2O basis changed")
    years = list(range(2022, 2100))
    regions = list(map(str, historical.reg_land.values))
    require(list(map(str, historical.bio_land.values)) == ORDER, "Historical biome order changed")
    pi_shift = np.asarray(historical.dA_shift_pi.values, dtype=np.float64)
    pi_wood = np.asarray(historical.dA_wharv_pi.values, dtype=np.float64)
    archive = xr.open_dataset(ARCHIVE)
    require(float(np.max(np.abs(pi_shift - np.asarray(archive.dA_shift_pi.values, dtype=np.float64)))) == 0.0,
            "Historical shift baseline differs from the OSCAR archive")
    require(float(np.max(np.abs(pi_wood - np.asarray(archive.dA_wharv_pi.values, dtype=np.float64)))) == 0.0,
            "Historical wood-harvest baseline differs from the OSCAR archive")
    archive.close()
    results = [build_one(key, spec, historical, years, regions, pi_shift, pi_wood) for key, spec in SCENARIOS.items()]
    ids = [int(value) for value in historical.source_config.values]
    manifest = {
        "statement": "These drivers were built in this study from the official LUH transitions, not taken from the age-module event_area variable.",
        "n2o_mass_basis": "TgN_two_nitrogen_atoms_v1",
        "years": [2022, 2099],
        "year_count": 78,
        "anchor_year": 2022,
        "integrated_years": [2023, 2099],
        "members": 500,
        "member_order_source": "historical_concentration_forcing.nc source_config, identical to outputs/oscar_origin/historical_concentration/final_state.nc",
        "source_config_first": ids[0],
        "source_config_last": ids[-1],
        "source_config": ids,
        "aggregation_rule": {
            "regional_reduction": "Existing v4 reduction of official LUH3 transitions: cell area in Mha times forest850 region weights, negative conversions clipped to zero, unavailable plantation stored as dormant zero. Scripts: workflows/prepare_land_use.py and workflows/prepare_future_land_use.py. This study does not read event_area.",
            "biome_map": "Forest=primf+secdf+pltns; Non-Forest=primn+secdn; Cropland=c3ann+c4ann+c3per+c4per+c3nfx; Pasture=pastr+range; Urban=urban. This is the map that reproduces the OSCAR archive, not the forest-age experiment map that placed range in Non-Forest.",
            "drivers": "D_dA_lcc1 is every conversion leaving primf or primn. D_dA_shift is the paired secondary exchange between Forest or Non-Forest and Cropland, min of the two directions after the regional sum. D_dA_lcc2 is the remaining secondary conversion, including within-cropland rotation. D_dA_wharv1 is primf and primn harvest. D_dA_wharv2 is secmf+secyf+pltns and secnf harvest.",
            "anomaly_convention": "Same as oscar format_LUH: lcc1, lcc2 and wharv1 stay absolute. shift and wharv2 are minus the 1750 baselines stored on the historical forcing (dA_shift_pi, dA_wharv_pi).",
            "time": "OSCAR year Y uses the LUH interval labelled Y. 2022 is kept as the anchor row. The age-module right-end shift, which zeroes that anchor, is not used.",
            "not_used": "native_background_land_only.nc was used only to check variable names and dimensions. Its H/M/VL values were not copied. L and ML have no official land file and are not built.",
        },
        "continuity_note": "2022 is compared with historical_concentration_forcing.nc. Off-diagonal cover change, shifting cultivation and wood harvest differ by at most a few tenths of Mha yr-1 because shifting is paired after the regional sum and the archive is float32. The cropland diagonal of D_dA_lcc2 is larger: this LUH3 contains within-crop transitions, especially c3nfx-c4ann in Africa, that the v4-beta2.1 archive does not. That difference is reported, not removed.",
        "scenarios": results,
    }
    path = OUT_DIR / "future_original_land_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({row["scenario"]: row["continuity_2022_against_historical_forcing"]["all_fields_max_abs_Mha_yr-1"] for row in results}))


if __name__ == "__main__":
    main()

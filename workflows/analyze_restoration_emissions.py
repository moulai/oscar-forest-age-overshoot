#!/usr/bin/env python3
"""Analyze the fixed restoration experiment without launching any integration.

Temperature is annual D_Tg minus each original member's 1850--1900 mean
from the common H1 emission-driven history. A return is the first post-peak
year at or below 1.5 K followed only by years at or below 1.5 K through
2099. These are window-persistent returns, not claims about later centuries.

All original members remain in the output. Duration comparisons condition
only on both arms having an overshoot and an interior peak; right-censored
members remain in that fixed set. Four-arm comparisons use its intersection.
Interval summaries are finite-ensemble identification bounds, not confidence
intervals. No resampling, weighting, significance test, or survival model is
used. The analysis does not import, call, or launch a model runner.

With --backgrounds, only the fixed A/B x on/off four-arm comparison is
required for each selected background. Every arm must retain all original
500 members. Omitting --backgrounds retains the full 280-cell analysis.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "outputs/age_module/restoration_emissions/PLAN.json"
SCHEMA = "restoration-overshoot-plan-v2"
SCENARIOS = ("H", "HL", "M", "VL", "LN")
STARTS = (2030, 2035, 2040, 2045, 2050, 2055, 2060)
SPANS = (5, 10, 15, 20)
MODES = ("on", "off")
FIRST_YEAR, LAST_YEAR = 2022, 2099
BASELINE_FIRST, BASELINE_LAST = 1850, 1900
THRESHOLD_K = 1.5
DURATION_THRESHOLD_YEARS = 5.0
EXPECTED_MEMBERS = 500
STATUSES = ("no_overshoot", "unresolved_peak", "returned",
            "right_censored", "missing")
DEFINED = frozenset(("returned", "right_censored"))
ARM_KEYS = ("A_on", "B_on", "A_off", "B_off")


class DependencyError(RuntimeError):
    """Required results or their declared identity are not available."""


@dataclass(frozen=True)
class Interval:
    """An interval; infinite limits mean no finite bound, never missing data."""

    lower: float
    upper: float
    lower_strict: bool = False
    upper_strict: bool = False

    def __post_init__(self):
        if (np.isnan(self.lower) or np.isnan(self.upper)
                or self.lower > self.upper
                or self.lower == np.inf or self.upper == -np.inf):
            raise ValueError("Invalid interval bounds")

    def as_dict(self):
        return {"lower": self.lower, "upper": self.upper,
                "lower_strict": self.lower_strict,
                "upper_strict": self.upper_strict}


@dataclass(frozen=True)
class Endpoint:
    status: str
    peak_K: Optional[float] = None
    peak_year: Optional[int] = None
    return_year: Optional[Interval] = None
    duration_years: Optional[Interval] = None

    @property
    def duration_defined(self):
        return self.status in DEFINED


def _years(values):
    years = np.asarray(values)
    if (years.ndim != 1 or years.size < 2
            or not np.all(np.isfinite(years))
            or not np.array_equal(years, years.astype(np.int64))
            or not np.all(np.diff(years) == 1)):
        raise ValueError("Years must be consecutive integers")
    return years.astype(np.int64)


def trajectory_endpoints(years, temperature) -> Endpoint:
    """Extract fixed-threshold endpoints from an already rebased trajectory.

    A tied maximum uses its last year. Endpoint maxima do not identify a peak.
    For a right-censored return, r > H and D > H - p. The numeric bounds are
    H and H - p, with strictness retained separately.
    """
    years = _years(years)
    temperature = np.asarray(temperature, dtype=float)
    if temperature.shape != years.shape:
        raise ValueError("Temperature and year shapes differ")
    if not np.all(np.isfinite(temperature)):
        return Endpoint("missing")
    peak = float(np.max(temperature))
    j = int(np.flatnonzero(temperature == peak)[-1])
    peak_year = int(years[j])
    if peak <= THRESHOLD_K:
        return Endpoint("no_overshoot", peak, peak_year)
    if j == 0 or j == len(years) - 1:
        return Endpoint("unresolved_peak", peak, peak_year)

    # The first year of the final below-threshold suffix is the window return.
    if temperature[-1] <= THRESHOLD_K:
        last_above = int(np.flatnonzero(temperature[j:] > THRESHOLD_K)[-1]) + j
        returned = int(years[last_above + 1])
        r = Interval(float(returned), float(returned))
        d = Interval(float(returned - peak_year), float(returned - peak_year))
        return Endpoint("returned", peak, peak_year, r, d)
    horizon = int(years[-1])
    return Endpoint("right_censored", peak, peak_year,
                    Interval(float(horizon), np.inf, lower_strict=True),
                    Interval(float(horizon - peak_year), np.inf, lower_strict=True))


def subtract_intervals(a: Interval, b: Interval) -> Interval:
    lower, upper = a.lower - b.upper, a.upper - b.lower
    return Interval(lower, upper,
                    bool(np.isfinite(lower) and (a.lower_strict or b.upper_strict)),
                    bool(np.isfinite(upper) and (a.upper_strict or b.lower_strict)))


def median_bounds(intervals: Sequence[Interval]) -> Optional[Interval]:
    """Conservative median envelope; never discard infinite bounds.

    NumPy's median uses the mean of the middle two values for even n. Lower
    bounds here can be -inf but not +inf; upper bounds can be +inf but not
    -inf, so this operation cannot create an inf + -inf ambiguity. Strict
    endpoint bounds are conservatively closed for aggregate decisions.
    """
    if not intervals:
        return None
    return Interval(float(np.median([v.lower for v in intervals])),
                    float(np.median([v.upper for v in intervals])))


def _at_least(interval: Optional[Interval], threshold: float):
    if interval is None:
        return "not_defined"
    if interval.lower >= threshold:
        return "supported"
    if interval.upper < threshold:
        return "not_reached"
    return "unidentified"


def _nonpositive(interval: Optional[Interval]):
    if interval is None:
        return "not_defined"
    if interval.upper <= 0:
        return "supported"
    if interval.lower > 0:
        return "not_supported"
    return "unidentified"


def _positive(interval: Optional[Interval]):
    if interval is None:
        return "not_defined"
    if interval.lower > 0:
        return "supported"
    if interval.upper <= 0:
        return "not_supported"
    return "unidentified"


def _ids(values):
    result = [value.item() if isinstance(value, np.generic) else value for value in values]
    if not result or len(set(result)) != len(result):
        raise ValueError("source_config must contain unique member identifiers")
    return result


def _status_counts(endpoints):
    n = len(endpoints)
    return {status: {"count": sum(v.status == status for v in endpoints),
                     "fraction_of_original_members": sum(v.status == status for v in endpoints) / n}
            for status in STATUSES}


def _metric(intervals):
    bound = median_bounds(intervals)
    return {"n": len(intervals),
            "median_bounds": None if bound is None else bound.as_dict()}


def _exact(value):
    return Interval(float(value), float(value))


def _flatten_interval(row, prefix, interval):
    for key in ("lower", "upper", "lower_strict", "upper_strict"):
        row[prefix + "_" + key] = None if interval is None else getattr(interval, key)


def _pair_metrics(a: Endpoint, b: Endpoint):
    dp = _exact(a.peak_year - b.peak_year)
    dr = subtract_intervals(a.return_year, b.return_year)
    dd = subtract_intervals(a.duration_years, b.duration_years)
    # Check the identity per member, not by subtracting ensemble medians.
    reconstructed = subtract_intervals(dr, dp)
    if dd.lower != reconstructed.lower or dd.upper != reconstructed.upper:
        raise AssertionError("Member identity delta_D = delta_r - delta_p failed")
    return {"delta_peak_K": _exact(a.peak_K - b.peak_K),
            "delta_peak_year": dp, "delta_return_year": dr,
            "delta_duration_years": dd}


def pair_analysis(a: Sequence[Endpoint], b: Sequence[Endpoint], source_config,
                  eligible_mask=None, population="S_pair"):
    """Return all-member rows and a labeled, censor-aware conditional summary."""
    source_config = _ids(source_config)
    n = len(source_config)
    if len(a) != n or len(b) != n:
        raise ValueError("Endpoint arrays must preserve the full member axis")
    eligible = np.array([x.duration_defined and y.duration_defined
                         for x, y in zip(a, b)], dtype=bool)
    if eligible_mask is not None:
        requested = np.asarray(eligible_mask, dtype=bool)
        if requested.shape != (n,) or np.any(requested & ~eligible):
            raise ValueError("Requested population includes an undefined duration")
        eligible = requested
    rows, collected = [], {key: [] for key in
                            ("delta_peak_K", "delta_peak_year", "delta_return_year",
                             "delta_duration_years")}
    complete_differences, definite, possible = [], 0, 0
    for position, (source, x, y, use) in enumerate(zip(source_config, a, b, eligible)):
        row = {"member_position": position, "source_config": source,
               "eligible": bool(use), "A_status": x.status, "B_status": y.status,
               "delta_peak_K_all_members": None if x.peak_K is None or y.peak_K is None
               else x.peak_K - y.peak_K}
        metrics = _pair_metrics(x, y) if use else {}
        for name in collected:
            _flatten_interval(row, name, metrics.get(name))
            if use:
                collected[name].append(metrics[name])
        if use:
            dd = metrics["delta_duration_years"]
            if x.peak_K < y.peak_K:
                definite += int(dd.lower >= DURATION_THRESHOLD_YEARS)
                possible += int(dd.upper >= DURATION_THRESHOLD_YEARS)
            if x.status == y.status == "returned":
                complete_differences.append(dd.lower)
        rows.append(row)
    count = int(eligible.sum())
    summary = {"population": population, "conditional_population": True,
               "n_original_members": n, "n_eligible": count,
               "eligible_source_config": [source_config[i] for i in np.flatnonzero(eligible)],
               "A_status_all_original_members": _status_counts(a),
               "B_status_all_original_members": _status_counts(b)}
    if any(x.status == "missing" for x in (*a, *b)):
        summary.update(status="blocked_missing", formal_summary_available=False)
        return rows, summary
    summary.update(status="complete", formal_summary_available=True)
    summary["peak_all_original_members"] = {
        "A_median_K": float(np.median([x.peak_K for x in a])),
        "B_median_K": float(np.median([x.peak_K for x in b])),
        "difference_of_medians_K": float(np.median([x.peak_K for x in a])
                                          - np.median([x.peak_K for x in b])),
        "median_of_paired_differences_K": float(np.median([x.peak_K - y.peak_K
                                                           for x, y in zip(a, b)])),
        "A_median_strictly_lower_than_B": bool(np.median([x.peak_K for x in a])
                                               < np.median([x.peak_K for x in b])),
        "interpretation": f"{FIRST_YEAR}-{LAST_YEAR} window maxima, including unresolved endpoint peaks"}
    summary["metrics"] = {name: _metric(values) for name, values in collected.items()}
    if not count:
        summary.update(information_gate="not_defined", duration_at_least_5_years="not_defined",
                       off_duration_nonpositive="not_defined", calendar_return="not_defined")
        return rows, summary
    ca = sum(use and x.status == "right_censored" for x, use in zip(a, eligible))
    cb = sum(use and x.status == "right_censored" for x, use in zip(b, eligible))
    summary["right_censoring_in_fixed_population"] = {
        "denominator": count, "A_count": ca, "B_count": cb,
        "A_fraction": ca / count, "B_fraction": cb / count}
    conditional_a = [x.peak_K for x, use in zip(a, eligible) if use]
    conditional_b = [x.peak_K for x, use in zip(b, eligible) if use]
    summary["peak_in_fixed_population"] = {
        "denominator": count, "A_median_K": float(np.median(conditional_a)),
        "B_median_K": float(np.median(conditional_b)),
        "difference_of_medians_K": float(np.median(conditional_a) - np.median(conditional_b)),
        "A_median_strictly_lower_than_B": bool(np.median(conditional_a) < np.median(conditional_b))}
    summary["information_gate"] = "not_met" if ca / count > .5 or cb / count > .5 else "met"
    dd = median_bounds(collected["delta_duration_years"])
    dr = median_bounds(collected["delta_return_year"])
    summary["duration_at_least_5_years"] = _at_least(dd, DURATION_THRESHOLD_YEARS)
    summary["off_duration_nonpositive"] = _nonpositive(dd)
    summary["calendar_return"] = (
        "later_calendar_return_identified" if dr.lower > 0 else
        "calendar_return_not_later_identified" if dr.upper <= 0 else "unidentified")
    summary["joint_lower_peak_and_duration_at_least_5_years"] = {
        "denominator": count, "definite_count": definite, "possible_count": possible,
        "definite_fraction": definite / count, "possible_fraction": possible / count}
    summary["complete_returners_description_only"] = {
        "n": len(complete_differences),
        "median_delta_duration_years": float(np.median(complete_differences))
        if complete_differences else None,
        "use": "Conditional description; not the primary median or a decision criterion"}
    return rows, summary


def four_arm_analysis(arms: Mapping[str, Sequence[Endpoint]], source_config):
    """Compare A=2030/span5 with B=2050/span5 and their on/off interaction."""
    source_config = _ids(source_config)
    if set(arms) != set(ARM_KEYS) or any(len(arms[k]) != len(source_config) for k in ARM_KEYS):
        raise ValueError("Four complete member axes A_on, B_on, A_off, B_off are required")
    common = np.array([all(arms[k][i].duration_defined for k in ARM_KEYS)
                       for i in range(len(source_config))])
    on_rows, on = pair_analysis(arms["A_on"], arms["B_on"], source_config, population="S_on")
    off_rows, off = pair_analysis(arms["A_off"], arms["B_off"], source_config, population="S_off")
    _, common_on = pair_analysis(arms["A_on"], arms["B_on"], source_config,
                                 common, population="S4")
    _, common_off = pair_analysis(arms["A_off"], arms["B_off"], source_config,
                                 common, population="S4")
    rows, interactions = [], []
    for i, source in enumerate(source_config):
        row = {"member_position": i, "source_config": source,
               "in_S_on": on_rows[i]["eligible"], "in_S_off": off_rows[i]["eligible"],
               "in_S4": bool(common[i])}
        for label, item in (("on", on_rows[i]), ("off", off_rows[i])):
            row.update({label + "_" + key: value for key, value in item.items()
                        if key not in ("source_config", "member_position", "eligible")})
        interaction = None
        if common[i]:
            d_on = _pair_metrics(arms["A_on"][i], arms["B_on"][i])["delta_duration_years"]
            d_off = _pair_metrics(arms["A_off"][i], arms["B_off"][i])["delta_duration_years"]
            interaction = subtract_intervals(d_on, d_off)
            interactions.append(interaction)
        _flatten_interval(row, "I_D_years", interaction)
        rows.append(row)
    summary = {"n_original_members": len(source_config),
               "S_on": on, "S_off": off,
               "S4": {"conditional_population": True, "n": int(common.sum()),
                      "source_config": [source_config[i] for i in np.flatnonzero(common)],
                      "on": common_on, "off": common_off}}
    if any(x.status == "missing" for k in ARM_KEYS for x in arms[k]):
        summary.update(status="blocked_missing", formal_summary_available=False)
        return rows, summary
    summary.update(status="complete", formal_summary_available=True)
    common_summary = summary["S4"]
    common_summary["I_D_years"] = _metric(interactions)
    common_summary["positive_interaction"] = _positive(median_bounds(interactions))
    if common.sum():
        off_dd = median_bounds([_pair_metrics(arms["A_off"][i], arms["B_off"][i])
                               ["delta_duration_years"] for i in np.flatnonzero(common)])
        off_peak_lower = common_off["peak_in_fixed_population"]["A_median_strictly_lower_than_B"]
        if off_dd.lower > 0 and off_peak_lower:
            common_summary["age_creation_interpretation"] = (
                "tradeoff_remains_off; age_creation_is_not_supported; "
                "a_positive_interaction_can_support_amplification")
        elif common_off["off_duration_nonpositive"] == "supported":
            common_summary["age_creation_interpretation"] = (
                "off_duration_condition_met; assess_on_peak_duration_and_information_gates_together")
        else:
            common_summary["age_creation_interpretation"] = "unidentified"
    else:
        common_summary["age_creation_interpretation"] = "not_defined"
    return rows, summary


def _check_member_axis(dataset: xr.Dataset, expected_source_config):
    if "source_config" not in dataset or dataset.source_config.dims != ("config",):
        raise DependencyError("A source_config(config) identity coordinate is required")
    expected = _ids(expected_source_config)
    actual = _ids(dataset.source_config.values)
    if actual != expected:
        raise DependencyError("source_config differs from the original member IDs or order")
    if "config" not in dataset.coords or not np.array_equal(dataset.config.values, np.arange(len(expected))):
        raise DependencyError("config must preserve the original positional axis 0..499")


def _temperature(dataset: xr.Dataset):
    if "D_Tg" not in dataset or set(dataset.D_Tg.dims) != {"year", "config"}:
        raise DependencyError("Annual D_Tg(year, config) is required")
    return dataset.D_Tg.transpose("year", "config")


def historical_reference(history: xr.Dataset, source_config):
    """Return each original member's H1 emissions offset and raw 2022 anchor.

    The caller must supply the declared emission-driven history. There is no
    concentration-driven fallback, pooled-member offset, or skip-NaN mean.
    """
    _check_member_axis(history, source_config)
    temperature = _temperature(history)
    try:
        years = _years(temperature.year.values)
    except ValueError as exc:
        raise DependencyError(str(exc)) from exc
    baseline_years = np.arange(BASELINE_FIRST, BASELINE_LAST + 1)
    if not np.all(np.isin(baseline_years, years)) or FIRST_YEAR not in years:
        raise DependencyError("H1 emissions history needs every 1850-1900 year and its 2022 anchor")
    baseline = temperature.sel(year=baseline_years).values.astype(float)
    anchor = temperature.sel(year=FIRST_YEAR).values.astype(float)
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(anchor)):
        raise DependencyError("H1 emissions baseline or 2022 anchor has missing member data")
    return baseline.mean(axis=0), anchor


def _analysis_years(limits):
    """Read the declared limits while requiring the fixed formal common window."""
    limits = np.asarray(limits)
    if limits.shape != (2,) or not np.array_equal(limits, [FIRST_YEAR, LAST_YEAR]):
        raise DependencyError(f"Formal PLAN years must be [{FIRST_YEAR}, {LAST_YEAR}]")
    return np.arange(int(limits[0]), int(limits[1]) + 1)


def rebase_future(future: xr.Dataset, source_config, offset, anchor,
                  analysis_years=(FIRST_YEAR, LAST_YEAR)):
    """Prepend the common 2022 anchor and subtract the same historical offset."""
    _check_member_axis(future, source_config)
    temperature = _temperature(future)
    years = _analysis_years(analysis_years)
    future_years = years[1:]
    if not np.array_equal(temperature.year.values, future_years):
        raise DependencyError(f"Future annual D_Tg must cover every year {future_years[0]}-{future_years[-1]}")
    offset, anchor = np.asarray(offset), np.asarray(anchor)
    if offset.shape != (len(source_config),) or anchor.shape != offset.shape:
        raise ValueError("Historical offset and anchor must retain the complete member axis")
    values = np.concatenate((anchor[None, :], temperature.values.astype(float)), axis=0)
    return years, values - offset[None, :]


def endpoint_rows(endpoints, source_config, cell):
    rows = []
    for position, (source, endpoint) in enumerate(zip(source_config, endpoints)):
        row = {name: cell[name] for name in ("scenario", "start", "span", "schedule", "age_production")}
        row.update(member_position=position, source_config=source, status=endpoint.status,
                   peak_K=endpoint.peak_K, peak_year=endpoint.peak_year)
        _flatten_interval(row, "return_year", endpoint.return_year)
        _flatten_interval(row, "duration_years", endpoint.duration_years)
        rows.append(row)
    return rows


def _read_plan(path: Path):
    if not path.is_file():
        raise DependencyError("Required PLAN is not available: " + str(path))
    plan = json.loads(path.read_text())
    if plan.get("schema") != SCHEMA:
        raise DependencyError("Analysis requires " + SCHEMA + "; the supplied plan schema is unsupported")
    _analysis_years(plan.get("years"))
    try:
        source = _ids(plan["source_config"])
        cells = plan["cells"]
        history = plan["history_directory"]
    except (KeyError, TypeError, ValueError) as exc:
        raise DependencyError("PLAN lacks its complete original member or history identity") from exc
    if len(source) != EXPECTED_MEMBERS:
        raise DependencyError("PLAN must preserve all 500 original source_config identifiers")
    expected = {(scenario, start, span, mode) for scenario in SCENARIOS
                for start in STARTS for span in SPANS for mode in MODES}
    try:
        keys = [(v["scenario"], v["start"], v["span"], v["age_production"]) for v in cells]
        outputs = [v["output"] for v in cells]
        schedules_match = all(v["schedule"] == f"start_{v['start']}_span_{v['span']}" for v in cells)
    except (KeyError, TypeError) as exc:
        raise DependencyError("PLAN cells lack required v2 analysis fields") from exc
    if (len(keys) != len(expected) or set(keys) != expected
            or len(set(outputs)) != len(outputs) or not schedules_match):
        raise DependencyError("PLAN must contain the fixed five-pathway 28 x on/off design exactly once")
    if not isinstance(history, str) or not history:
        raise DependencyError("PLAN history_directory is not defined")
    return plan, source, cells


def _project_path(root: Path, path):
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise DependencyError("PLAN dependency is outside the study directory")
    return candidate


def _json_safe(value):
    """Write strict JSON: infinity is explicit text; undefined is null."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            raise ValueError("NaN is not an analysis result")
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty analysis table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(value) for key, value in row.items()})


def _selected_backgrounds(backgrounds):
    if backgrounds is None:
        return SCENARIOS
    requested = tuple(backgrounds)
    if (not requested or len(set(requested)) != len(requested)
            or any(value not in SCENARIOS for value in requested)):
        raise ValueError("Backgrounds must be a nonempty unique subset of H, HL, M, VL, LN")
    return tuple(value for value in SCENARIOS if value in requested)


def analyze_plan(plan_path: Path, output: Path, root: Path = ROOT, backgrounds=None):
    """Analyze a fixed scope; a missing selected dependency writes nothing.

    An explicit background selection always means A=2030/span5 and
    B=2050/span5, each on/off. It never selects individual cells or members.
    """
    plan_path, output, root = Path(plan_path), Path(output), Path(root)
    plan, source, matrix_cells = _read_plan(plan_path)
    selected = _selected_backgrounds(backgrounds)
    cells = matrix_cells if backgrounds is None else [
        cell for cell in matrix_cells if cell["scenario"] in selected
        and cell["start"] in (2030, 2050) and cell["span"] == 5]
    history_path = _project_path(root, plan["history_directory"]) / "annual_summary.nc"
    paths = [_project_path(root, cell["output"]) / "annual_summary.nc" for cell in cells]
    missing = [path for path in (history_path, *paths) if not path.is_file()]
    if missing:
        examples = "; ".join(str(path) for path in missing[:4])
        raise DependencyError(f"{len(missing)} required result files are not available: {examples}")
    if output.exists():
        raise DependencyError("Analysis output already exists; choose a new output directory: " + str(output))
    with xr.open_dataset(history_path) as history:
        offset, anchor = historical_reference(history, source)
    endpoints_by_key, all_rows, cell_summaries = {}, [], []
    for cell, path in zip(cells, paths):
        with xr.open_dataset(path) as future:
            years, temperature = rebase_future(future, source, offset, anchor,
                                                analysis_years=plan["years"])
        endpoints = [trajectory_endpoints(years, temperature[:, i]) for i in range(len(source))]
        key = (cell["scenario"], cell["start"], cell["span"], cell["age_production"])
        endpoints_by_key[key] = endpoints
        all_rows.extend(endpoint_rows(endpoints, source, cell))
        cell_summaries.append({**{name: cell[name] for name in
                                 ("scenario", "start", "span", "schedule", "age_production")},
                               "status_all_original_members": _status_counts(endpoints)})
    blocked = any(row["status"] == "missing" for row in all_rows)
    comparisons, pair_rows = {}, []
    for scenario in selected:
        arms = {"A_on": endpoints_by_key[(scenario, 2030, 5, "on")],
                "B_on": endpoints_by_key[(scenario, 2050, 5, "on")],
                "A_off": endpoints_by_key[(scenario, 2030, 5, "off")],
                "B_off": endpoints_by_key[(scenario, 2050, 5, "off")]}
        rows, summary = four_arm_analysis(arms, source)
        pair_rows.extend({"scenario": scenario, **row} for row in rows)
        if not blocked:
            comparisons[scenario] = summary
    matrix_present = sum((_project_path(root, cell["output"]) / "annual_summary.nc").is_file()
                         for cell in matrix_cells)
    protocol = {
        "schema": "restoration-overshoot-analysis-v1", "plan": str(plan_path),
        "plan_schema": SCHEMA, "history": str(history_path),
        "input_manifest_sha256": plan.get("input_manifest_sha256"),
        "history_run_identity": plan.get("history_run_identity"),
        "ordered_source_ids_sha256": plan.get("ordered_source_ids_sha256"),
        "source_config": source, "n_original_members": len(source),
        "reference": "Per-member 1850-1900 mean of the common H1 emissions history",
        "baseline_years_inclusive": [BASELINE_FIRST, BASELINE_LAST],
        "temperature_years_inclusive": list(plan["years"]),
        "threshold_K": THRESHOLD_K, "annual_rule": "integer year, no interpolation or smoothing",
        "peak_tie_rule": "last maximum; a selected peak at either endpoint is unresolved",
        "return_rule": f"first post-peak year <= threshold with all subsequent years through {plan['years'][-1]} <= threshold",
        "right_censor_rule": f"r > {plan['years'][-1]}; lower={plan['years'][-1]} with strict flag, upper=+Infinity",
        "duration_effect_threshold_years": DURATION_THRESHOLD_YEARS,
        "information_gate": "More than 50% right-censored in either arm of the fixed definable population is not met; not a refutation",
        "conditional_populations": "S_on and S_off require two definable overshoots; S4 their intersection; right-censored members remain",
        "median_rule": "mean of middle two when even; median lower/upper bounds retain infinity and conservatively close strict bounds",
        "date_identity": "delta_D = delta_r - delta_p per member; not an arithmetic identity between medians",
        "inference": "Descriptive original finite ensemble; no resampling, member weighting, p-values, or real-world probabilities",
        "main_comparison": {"A": {"start": 2030, "span": 5}, "B": {"start": 2050, "span": 5},
                            "primary_paths": ["VL", "LN"], "background_reference": "M",
                            "grid_use": "range only; no selection of a winner"},
        "selection": {"mode": "full_matrix" if backgrounds is None else "fixed_primary_four_arms",
                      "backgrounds": list(selected), "selected_cell_count": len(cells),
                      "rule": "All prescribed 28 x on/off cells per background" if backgrounds is None else
                      "Declared backgrounds only; A=2030/span5 and B=2050/span5 x on/off are fixed before results",
                      "members": "All 500 original IDs in original order; no partial-member statistics"},
        "matrix_progress_at_analysis": {"expected_cell_count": len(matrix_cells),
                                        "annual_files_present": matrix_present,
                                        "all_annual_files_present": matrix_present == len(matrix_cells),
                                        "full_matrix_analysis_complete": backgrounds is None and not blocked},
        "combined_VL_LN_primary_report_complete": {"VL", "LN"}.issubset(selected) and not blocked,
        "final_grid_requirement": "The full prescribed 280-cell matrix is required for the final five-background range assessment; core-subset outputs do not replace it",
        "background_aggregation": "Comparisons are reported separately for each background; members are never pooled across backgrounds",
        "missing_data_rule": "All selected member rows retained; any missing trajectory blocks the selected scope's formal summaries",
        "infinite_json_values": "Explicit strings +Infinity and -Infinity; null means undefined",
        "emissions_integral": "Not calculated by this temperature endpoint analysis",
        "formal_summary_available": not blocked,
        "status": "blocked_missing" if blocked else "complete",
        "cell_statuses": cell_summaries, "main_comparisons": comparisons,
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "member_endpoints.csv", all_rows)
    _write_csv(output / "paired_members.csv", pair_rows)
    _write_csv(output / "historical_temperature_reference.csv", [
        {"member_position": i, "source_config": member,
         "D_Tg_1850_1900_mean_K": float(offset[i]), "D_Tg_2022_raw_K": float(anchor[i])}
        for i, member in enumerate(source)])
    (output / "summary.json").write_text(json.dumps(_json_safe(protocol), indent=2, allow_nan=False) + "\n")
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--backgrounds", nargs="+", choices=SCENARIOS,
                        help="Analyze only the fixed A/B x on/off four arms per named background")
    parser.add_argument("--output", type=Path,
                        help="New output directory; default: analysis/full_grid or analysis/primary_<backgrounds>")
    args = parser.parse_args(argv)
    try:
        selected = _selected_backgrounds(args.backgrounds)
        name = "full_grid" if args.backgrounds is None else "primary_" + "_".join(selected)
        output = args.output or ROOT / "outputs/age_module/restoration_emissions/analysis" / name
        result = analyze_plan(args.plan, output, backgrounds=args.backgrounds)
    except (DependencyError, OSError, ValueError, KeyError) as exc:
        print("DEPENDENCY NOT READY: " + str(exc), file=sys.stderr)
        return 3
    print(json.dumps({"status": result["status"], "output": str(output),
                      "formal_summary_available": result["formal_summary_available"]}))
    return 4 if not result["formal_summary_available"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

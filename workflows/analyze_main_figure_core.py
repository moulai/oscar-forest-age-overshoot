#!/usr/bin/env python3
"""Analyze climate outcomes and on-calendar land uptake across matched futures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

SCENARIOS = ('H', 'HL', 'M', 'VL', 'LN')
STAGES = ('before_upcrossing', 'upcrossing_to_maximum',
          'post_peak_before_return', 'after_return_to_2099')
STATES = ('never_above', 'exceeded_returned', 'exceeded_still_above')
THRESHOLD_K = 1.5


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def jwrite(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def describe(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, q05=None, median=None, q95=None,
                    negative=0, zero=0, positive=0)
    q = np.quantile(values, [.05, .5, .95])
    return dict(n=len(values), mean=values.mean(), q05=q[0], median=q[1], q95=q[2],
                min=values.min(), max=values.max(), negative=(values < 0).sum(),
                zero=(values == 0).sum(), positive=(values > 0).sum())


def endpoints(temperature, years):
    above = temperature > THRESHOLD_K
    peak = float(temperature.max())
    pi = int(np.flatnonzero(temperature == peak)[-1])
    any_above = bool(above.any())
    resolved = any_above and 0 < pi < len(years) - 1
    if not any_above:
        state, u, r = 'never_above', np.nan, np.nan
    else:
        u = int(years[np.flatnonzero(above)[0]])
        if above[-1]:
            state, r = 'exceeded_still_above', np.nan
        else:
            state = 'exceeded_returned'
            r = int(years[np.flatnonzero(above)[-1] + 1])
    return dict(window_max_K=peak, window_max_year=int(years[pi]),
                resolved_overshoot_peak=resolved,
                maximum_at_left_endpoint=pi == 0,
                maximum_at_right_endpoint=pi == len(years) - 1,
                state=state, upcrossing_year=u, return_year=r,
                upcrossing_left_truncated=bool(above[0]),
                return_right_censored=state == 'exceeded_still_above',
                exceedance_years_77=int(above[1:].sum()),
                burden_77_Kyr=float(np.maximum(temperature[1:] - THRESHOLD_K, 0).sum()),
                T2099_K=float(temperature[-1]))


def stage_masks(on_endpoint, future_years):
    u = on_endpoint['upcrossing_year']
    p = on_endpoint['window_max_year']
    r = on_endpoint['return_year']
    masks = [np.zeros(len(future_years), bool) for _ in STAGES]
    supports = ['not_applicable'] * 4
    if not np.isfinite(u):
        masks[0][:] = True
        supports[0] = 'upcrossing_unobserved_to_2099'
    else:
        masks[0] = future_years < u
        supports[0] = ('left_truncated_empty' if on_endpoint['upcrossing_left_truncated']
                       else 'observed_complete')
        if not on_endpoint['resolved_overshoot_peak']:
            masks[1] = future_years >= u
            supports[1] = 'maximum_unresolved_observed_part'
        else:
            masks[1] = (future_years >= u) & (future_years <= p)
            supports[1] = ('left_truncated' if on_endpoint['upcrossing_left_truncated']
                           else 'observed_complete')
            if np.isfinite(r):
                masks[2] = (future_years > p) & (future_years < r)
                supports[2] = 'observed_complete'
                masks[3] = future_years >= r
                supports[3] = 'observed_to_window_end'
            else:
                masks[2] = future_years > p
                supports[2] = 'return_censored_observed_part'
    require(np.array_equal(np.stack(masks).sum(axis=0), np.ones(len(future_years))),
            'Observed stage masks must partition all future annual intervals')
    return masks, supports


def analyze(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(source) as opened:
        ds = opened.load()
    years = ds.year.values.astype(int)
    fy = ds.future_year.values.astype(int)
    ids = ds.source_config.values
    require(tuple(ds.scenario.values) == SCENARIOS, 'Expected the five matched pathways')
    require(np.array_equal(years, np.arange(2022, 2100)), 'Expected 2022 anchor and 77 future means')
    require(np.array_equal(fy, years[1:]), 'Future intervals differ from temperature support')
    require(len(ids) == 500 and len(np.unique(ids)) == 500, 'Original 500 ordered IDs required')
    members, phases, annual, transitions, annual_members = [], [], [], [], []
    scenario_summary = {}

    for bg in SCENARIOS:
        g = ds.sel(scenario=bg)
        t = {a: g.T_1850_1900_K.sel(production=a).transpose('year', 'config').values for a in ('on', 'off')}
        co2 = {a: g.CO2_anomaly_ppm.sel(production=a).transpose('year', 'config').values for a in ('on', 'off')}
        net = {a: g.net_land_uptake_PgC_yr.sel(production=a).transpose('future_year', 'config').values for a in ('on', 'off')}
        npp = {a: g.NPP_PgC_yr.sel(production=a).transpose('future_year', 'config').values for a in ('on', 'off')}
        require(all(np.isfinite(v).all() for d in (t, co2, net, npp) for v in d.values()),
                'Undefined member observations in ' + bg)
        require(np.array_equal(t['on'][0], t['off'][0]), 'The common temperature anchor differs')
        cumulative = {a: np.vstack([np.zeros((1, 500)), np.cumsum(net[a], axis=0)]) for a in ('on', 'off')}
        start_m, start_p = len(members), len(phases)
        phase_code = np.full((len(years), 500), -1, int)

        for pos, source_id in enumerate(ids):
            ep = {a: endpoints(t[a][:, pos], years) for a in ('on', 'off')}
            row = dict(scenario=bg, member_position=pos, source_config=int(source_id))
            for a in ('on', 'off'):
                row.update({f'{a}_{k}': v for k, v in ep[a].items()})
                row[a + '_cumulative_net_2023_2099_PgC'] = cumulative[a][-1, pos]
            for key in ('window_max_K', 'window_max_year', 'exceedance_years_77',
                        'burden_77_Kyr', 'T2099_K', 'cumulative_net_2023_2099_PgC'):
                row['delta_' + key] = row['on_' + key] - row['off_' + key]
            row['both_resolved_overshoot_peaks'] = ep['on']['resolved_overshoot_peak'] and ep['off']['resolved_overshoot_peak']
            row['lost_return_by_2099'] = ep['off']['state'] == 'exceeded_returned' and ep['on']['state'] == 'exceeded_still_above'
            row['gained_return_by_2099'] = ep['on']['state'] == 'exceeded_returned' and ep['off']['state'] == 'exceeded_still_above'
            row['delta_return_year_exact'] = (ep['on']['return_year'] - ep['off']['return_year']
                                               if ep['on']['state'] == ep['off']['state'] == 'exceeded_returned' else np.nan)
            dc = cumulative['on'][:, pos] - cumulative['off'][:, pos]
            worst = float(dc.min())
            wi = int(np.flatnonzero(dc == worst)[-1])
            row['minimum_cumulative_net_difference_PgC'] = worst
            row['minimum_cumulative_net_difference_year'] = int(years[wi])
            row['recovery_from_minimum_to_2099_PgC'] = float(dc[-1] - worst)
            row['recovered_fraction_of_maximum_observed_deficit'] = (float((dc[-1] - worst) / -worst)
                                                                      if worst < 0 else np.nan)
            row['maximum_observed_deficit_defined'] = worst < 0
            members.append(row)
            masks, supports = stage_masks(ep['on'], fy)
            for si, (mask, support) in enumerate(zip(masks, supports)):
                phase_code[1:, pos][mask] = si
                count = int(mask.sum())
                pr = dict(scenario=bg, member_position=pos, source_config=int(source_id),
                          stage_index=si, stage=STAGES[si], stage_support=support,
                          stage_has_intervals=count > 0, duration_years=count,
                          first_annual_label=int(fy[mask][0]) if count else np.nan,
                          last_annual_label=int(fy[mask][-1]) if count else np.nan,
                          on_upcrossing_year=ep['on']['upcrossing_year'],
                          on_maximum_year=ep['on']['window_max_year'],
                          on_resolved_peak=ep['on']['resolved_overshoot_peak'],
                          on_return_year=ep['on']['return_year'],
                          on_upcrossing_left_truncated=ep['on']['upcrossing_left_truncated'],
                          stage_complete=support in ('observed_complete', 'observed_to_window_end'))
                for a in ('on', 'off'):
                    c = float(net[a][mask, pos].sum())
                    pr[a + '_stage_cumulative_net_PgC'] = c
                    pr[a + '_stage_mean_net_PgC_yr'] = c / count if count else np.nan
                pr['delta_stage_cumulative_net_PgC'] = pr['on_stage_cumulative_net_PgC'] - pr['off_stage_cumulative_net_PgC']
                pr['delta_stage_mean_net_PgC_yr'] = pr['delta_stage_cumulative_net_PgC'] / count if count else np.nan
                phases.append(pr)

        gm = pd.DataFrame(members[start_m:])
        gp = pd.DataFrame(phases[start_p:])
        for off_state in STATES:
            for on_state in STATES:
                n = int(((gm.off_state == off_state) & (gm.on_state == on_state)).sum())
                transitions.append(dict(scenario=bg, off_state=off_state, on_state=on_state,
                                        count=n, percent_of_original=100 * n / 500))
        for yi, year in enumerate(years):
            ar = dict(scenario=bg, year=int(year), n_original=500)
            fields = dict(on_temperature_K=t['on'][yi], off_temperature_K=t['off'][yi],
                          delta_temperature_K=t['on'][yi] - t['off'][yi],
                          on_CO2_ppm=co2['on'][yi], off_CO2_ppm=co2['off'][yi],
                          delta_CO2_ppm=co2['on'][yi] - co2['off'][yi],
                          on_cumulative_net_PgC=cumulative['on'][yi],
                          off_cumulative_net_PgC=cumulative['off'][yi],
                          delta_cumulative_net_PgC=cumulative['on'][yi] - cumulative['off'][yi])
            if yi:
                for a in ('on', 'off'):
                    fields[a + '_net_PgC_yr'] = net[a][yi - 1]
                    fields[a + '_NPP_PgC_yr'] = npp[a][yi - 1]
                fields['delta_net_PgC_yr'] = net['on'][yi - 1] - net['off'][yi - 1]
            for name, values in fields.items():
                ar.update({name + '_' + k: v for k, v in describe(values).items()})
            annual.append(ar)
            for pos, source_id in enumerate(ids):
                raw = dict(scenario=bg, member_position=pos, source_config=int(source_id),
                           year=int(year), on_stage_index=int(phase_code[yi, pos]))
                raw.update({k: float(v[pos]) for k, v in fields.items()})
                annual_members.append(raw)
        scenario_summary[bg] = dict(
            n_original=500, n_both_resolved_overshoot_peaks=int(gm.both_resolved_overshoot_peaks.sum()),
            delta_maximum_K=describe(gm.delta_window_max_K),
            delta_exceedance_years=describe(gm.delta_exceedance_years_77),
            delta_burden_Kyr=describe(gm.delta_burden_77_Kyr),
            delta_T2099_K=describe(gm.delta_T2099_K),
            cumulative_net_difference_PgC=describe(gm.delta_cumulative_net_2023_2099_PgC),
            CO2_difference_2099_ppm=describe(co2['on'][-1] - co2['off'][-1]),
            recovery_PgC=describe(gm.recovery_from_minimum_to_2099_PgC),
            recovered_fraction=describe(gm.recovered_fraction_of_maximum_observed_deficit),
            deficit_minimum_year=describe(gm.minimum_cumulative_net_difference_year),
            lost_return_n=int(gm.lost_return_by_2099.sum()),
            gained_return_n=int(gm.gained_return_by_2099.sum()),
            on_states={s: int((gm.on_state == s).sum()) for s in STATES},
            off_states={s: int((gm.off_state == s).sum()) for s in STATES},
            phases={})
        for si, name in enumerate(STAGES):
            ps = gp.loc[(gp.stage_index == si) & gp.stage_has_intervals]
            scenario_summary[bg]['phases'][name] = dict(
                n_nonempty=len(ps), n_complete=int(ps.stage_complete.sum()),
                support_counts={str(k): int(v) for k, v in ps.stage_support.value_counts().items()},
                metrics={k: describe(ps[k]) for k in ('duration_years', 'on_stage_mean_net_PgC_yr',
                         'off_stage_mean_net_PgC_yr', 'delta_stage_mean_net_PgC_yr',
                         'on_stage_cumulative_net_PgC', 'off_stage_cumulative_net_PgC',
                         'delta_stage_cumulative_net_PgC')})

    pd.DataFrame(members).to_csv(output / 'member_outcomes_all500.csv', index=False)
    pd.DataFrame(phases).to_csv(output / 'member_stage_carbon_all500.csv', index=False)
    pd.DataFrame(annual).to_csv(output / 'annual_summary.csv', index=False)
    pd.DataFrame(transitions).to_csv(output / 'state_transitions.csv', index=False)
    pd.DataFrame(annual_members).to_csv(output / 'annual_members_all500.csv.gz', index=False, compression='gzip')
    phase_rows = []
    for bg in SCENARIOS:
        for si, name in enumerate(STAGES):
            v = scenario_summary[bg]['phases'][name]
            rr = dict(scenario=bg, stage_index=si, stage=name,
                      n_original=500, n_nonempty=v['n_nonempty'], n_complete=v['n_complete'])
            for metric, values in v['metrics'].items():
                rr.update({metric + '_' + k: vv for k, vv in values.items()})
            phase_rows.append(rr)
    pd.DataFrame(phase_rows).to_csv(output / 'stage_summary.csv', index=False)
    metadata = dict(
        schema='main-figure-climate-land-core-v1', status='complete',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source.resolve()), source_bytes=source.stat().st_size,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_attributes=dict(ds.attrs), original_source_config=ids.tolist(),
        family='Five fully matched CMIP7 futures; unretimed land activity; shared H1-2022 full state',
        threshold_K=THRESHOLD_K, window=[2022, 2099], integral_years=[2023, 2099],
        member_count_per_condition=500, stage_calendar='Each on member defines stages; its off member uses exactly those years',
        stage_rule='y<u; u<=y<=p; p<y<r; y>=r. Unresolved peak / return: observed partial segment, not imputed event.',
        net_definition='D_Fland-D_Eluc; all native land classes/pools excluding permafrost; positive uptake anomaly',
        uncertainty='Within-condition empirical original-parameter distribution; not a confidence interval or real-world probability',
        stage_empty_rule='Zero observed intervals: cumulative observed contribution zero, average undefined; phase absent flag retained',
        scenarios=scenario_summary)
    jwrite(output / 'summary.json', metadata)
    shutil.copy2(__file__, output / 'analysis_source_used.py')
    (output / 'README.md').write_text(
        '# Main-figure climate and land-uptake core\n\n'
        'Five fully matched pathways, original land-activity timing, common H1 end-2022 state. '
        'All 500 ordered member IDs are retained per pathway.\n\n'
        'member_outcomes_all500.csv: paired maxima, threshold years, window-return states and flags. '
        'member_stage_carbon_all500.csv: each on-defined calendar stage and both uptake trajectories, '
        'including empty and incomplete-stage flags. annual_members_all500.csv.gz: all original annual '
        'temperatures, CO2 and land uptake with cumulative uptake. Stage durations vary by member; '
        'stage medians cannot be added as a budget. The all-year cumulative series retains every member.\n\n'
        'Annual interval y is (y-1,y]. Integrals start in 2023; 2022 is only an anchor. '
        'A return is the first year of a final T<=1.5 K suffix through 2099. Initial above-threshold '
        'members are left-truncated; endpoint maxima are not resolved physical peaks. '
        'Incomplete phases are observed partial sums, not completed phase totals. '
        'The land quantity is D_Fland-D_Eluc, an all-land uptake anomaly excluding permafrost, '
        'not an age-resolved forest NBP.\n')
    print(json.dumps(clean({bg: {k: v for k, v in val.items() if k != 'phases'}
                            for bg, val in scenario_summary.items()}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)


if __name__ == '__main__':
    main()

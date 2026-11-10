#!/usr/bin/env python3
"""Summarize completed, independently spun-up H0/H1 historical experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


AGE = ['0_to_5', '5_to_100', 'over_100', 'primary_unaged']
FIELDS = {
    'temperature_raw_K': ('D_Tg', 'K'),
    'co2_anomaly_ppm': ('D_CO2', 'ppm'),
    'live_carbon_PgC': ('Cveg', 'PgC'),
    'cwd_carbon_PgC': ('Ccwd', 'PgC'),
    'soil_carbon_PgC': ('Csoil', 'PgC'),
    'product_carbon_PgC': ('Chwp', 'PgC'),
    'forest_area_Mha': ('rf_forest_area_mean', 'Mha'),
    'forest_wood_carbon_PgC': ('rf_forest_W_mean', 'PgC'),
    'forest_fine_carbon_PgC': ('rf_forest_F_mean', 'PgC'),
    'background_land_uptake_PgC_yr': ('D_Fland', 'PgC yr-1'),
    'land_use_emissions_PgC_yr': ('D_Eluc', 'PgC yr-1'),
    'npp_PgC_yr': ('rf_NPP', 'PgC yr-1'),
}


def dist(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (500,) or not np.isfinite(a).all():
        raise ValueError('Every paired summary requires the full 500 finite members.')
    q = np.quantile(a, [.05, .25, .5, .75, .95])
    return dict(n=500, p05=float(q[0]), p25=float(q[1]), median=float(q[2]),
                p75=float(q[3]), p95=float(q[4]), mean=float(a.mean()),
                negative=int((a < 0).sum()), zero=int((a == 0).sum()),
                positive=int((a > 0).sum()))


def extract(root, cache):
    sources, arrays, ordered_ids = [], [], None
    for mode in ('emissions', 'concentration'):
        arms = []
        for arm in ('H0', 'H1'):
            folder = Path('outputs/age_module') / ('historical_' + mode) / arm
            report = json.loads((root / folder / 'report.json').read_text())
            if report['status'] != 'passed_historical' or report['members'] != 500:
                raise ValueError(f'Historical formal result is incomplete: {folder}')
            path = root / folder / 'annual_summary.nc'
            with xr.open_dataset(path) as raw:
                ids = np.asarray(raw.source_config.values)
                if ordered_ids is None:
                    ordered_ids = ids.copy()
                if not np.array_equal(ids, ordered_ids):
                    raise ValueError('Historical original-member order differs between arms.')
                if not np.array_equal(raw.year.values, np.arange(1750, 2023)):
                    raise ValueError('Expected historical initial label 1750, then 1751..2022.')
                data = {}
                for name, (native, unit) in FIELDS.items():
                    value = raw[native]
                    dims = [d for d in value.dims if d not in ('year', 'config')]
                    value = value.sum(dims) if dims else value
                    data[name] = value.transpose('year', 'config').load()
                    data[name].attrs = dict(units=unit, native_variable=native)
                data['forest_age_area_Mha'] = raw.rf_age_area_mean.sel(age_support=AGE).sum('reg_land').transpose('year', 'config', 'age_support').load()
                data['forest_age_area_Mha'].attrs = dict(units='Mha', native_variable='rf_age_area_mean', primary='Unaged primary forest; no invented age.')
                ds = xr.Dataset(data).assign_coords(config=np.arange(500), source_config=('config', ids))
                ds['forest_live_carbon_PgC'] = ds.forest_wood_carbon_PgC + ds.forest_fine_carbon_PgC
                ds['net_land_uptake_PgC_yr'] = ds.background_land_uptake_PgC_yr - ds.land_use_emissions_PgC_yr
                ds['net_land_uptake_PgC_yr'].attrs = dict(units='PgC yr-1', definition='Global native D_Fland minus D_Eluc, positive atmosphere-to-land; all native land pools, not forest NPP.')
                ds['land_carbon_PgC'] = ds.live_carbon_PgC + ds.cwd_carbon_PgC + ds.soil_carbon_PgC + ds.product_carbon_PgC
                ds['temperature_PI_K'] = ds.temperature_raw_K - ds.temperature_raw_K.sel(year=slice(1850, 1900)).mean('year')
                ds['temperature_PI_K'].attrs = dict(units='K', reference='Each historical arm and each original member uses its own 1850..1900 annual mean.')
                # The first saved label is the initial value, not an integrated year.
                value = ds.net_land_uptake_PgC_yr.copy()
                value.loc[dict(year=1750)] = 0.
                ds['cumulative_net_uptake_since_1750_PgC'] = value.cumsum('year')
                ds['cumulative_net_uptake_since_1750_PgC'].attrs = dict(units='PgC', definition='Sum of native annual interval means for labels 1751..y, one year each; initial label 1750 contributes zero.')
                arms.append(ds.expand_dims(production=[arm]))
                sources.append(dict(mode=mode, arm=arm, annual_path=str(folder/'annual_summary.nc'),
                                    run_identity=report['run_identity'], spinup_identity=report.get('spinup_identity'),
                                    configuration=report.get('configuration'),
                                    input_manifest=report.get('input_manifest'),
                                    secondary_extension=report.get('secondary_extension'),
                                    annual_bytes=path.stat().st_size,
                                    time_sampling=raw.attrs.get('time_sampling')))
        arrays.append(xr.concat(arms, dim='production').expand_dims(driver_mode=[mode]))
    ds = xr.concat(arrays, dim='driver_mode')
    ds.attrs = dict(schema='ed03-historical-paired-all500-v1',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    member_policy='Original 500 ordered source IDs; all retained; no sampling or weighting.',
                    experiment='H0 and H1 include their own dynamic prehistories and histories; this is not a future switch at a common H1-2022 state.',
                    H0='Native annual cohorts and inventory-dependent land events are retained; age-production response is off (g_W=g_F=1). Not original OSCAR.',
                    H1='Age-production response on, with evolving annual cohorts.',
                    time_support='1750 initial value; 1751..2022 duration-weighted native interval means.',
                    provenance_json=json.dumps(sources, ensure_ascii=True))
    encoding = {k: dict(zlib=True, complevel=2) for k in ds.data_vars}
    ds.to_netcdf(cache, encoding=encoding)
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path('.'))
    ap.add_argument('--output', type=Path, default=Path('outputs/age_module/main_figure_redesign/figures/ed03_history_age_response/source_data'))
    ap.add_argument('--refresh-source', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output/'source_history_paired_all500.nc'
    ds = extract(args.project_root, cache) if args.refresh_source or not cache.exists() else xr.load_dataset(cache)
    ids = ds.source_config.values.astype(int)
    rows, endpoints, yearly, ages, reference_rows = [], [], [], [], []
    differences = ds.sel(production='H1', drop=True) - ds.sel(production='H0', drop=True)
    for mode in ds.driver_mode.values:
        d = ds.sel(driver_mode=mode)
        delta = differences.sel(driver_mode=mode)
        for name in [n for n in ds.data_vars if 'age_support' not in ds[n].dims]:
            for production in ('H0','H1','H1_minus_H0'):
                a = delta[name] if production == 'H1_minus_H0' else d[name].sel(production=production)
                for year in (1750, 1850, 1900, 1950, 2000, 2022):
                    v = a.sel(year=year).values
                    rows.append(dict(driver_mode=str(mode), production=production, year=year, metric=name, **dist(v)))
                for year, v in zip(ds.year.values, a.values):
                    yearly.append(dict(driver_mode=str(mode), production=production, year=int(year), metric=name, **dist(v)))
                v = a.sel(year=2022).values
                for i, value in enumerate(v):
                    endpoints.append(dict(driver_mode=str(mode), production=production, year=2022, member_position=i,
                                          source_config=int(ids[i]), metric=name, value=float(value)))
        # Preserve the complete paired raw/PI baseline decomposition per member.
        for i, identifier in enumerate(ids):
            rows_ref = dict(driver_mode=str(mode), member_position=i, source_config=int(identifier),
                            H0_PI_offset_K=float(d.temperature_raw_K.sel(production='H0', year=slice(1850,1900)).mean('year').values[i]),
                            H1_PI_offset_K=float(d.temperature_raw_K.sel(production='H1', year=slice(1850,1900)).mean('year').values[i]))
            reference_rows.append(rows_ref)
        for production in ('H0','H1'):
            area = d.forest_age_area_Mha.sel(production=production)
            share = area/area.sum('age_support')*100
            for support in AGE:
                for year in ds.year.values:
                    ages.append(dict(driver_mode=str(mode), production=production, year=int(year), age_support=support,
                                     **{'area_'+k:v for k,v in dist(area.sel(year=year,age_support=support).values).items()},
                                     **{'share_'+k:v for k,v in dist(share.sel(year=year,age_support=support).values).items()}))
    pd.DataFrame(rows).to_csv(args.output/'historical_selected_year_summaries.csv', index=False)
    pd.DataFrame(endpoints).to_csv(args.output/'historical_2022_members_all500.csv', index=False)
    pd.DataFrame(yearly).to_csv(args.output/'historical_annual_summaries.csv', index=False)
    pd.DataFrame(ages).to_csv(args.output/'historical_age_area_summaries.csv', index=False)
    pd.DataFrame(reference_rows).to_csv(args.output/'historical_temperature_references_all500.csv', index=False)
    summary = dict(status='complete', schema='ed03-historical-age-response-v1', years=[1750,2022],
                   original_member_count=500, driver_modes=['emissions','concentration'],
                   production_modes=['H0','H1'], primary_figure_driver='emissions',
                   paired_delta='H1 minus H0, within the same driver mode and original source_config before summarization.',
                   temperature_references='Both unrecentered D_Tg and own-history, own-member 1850..1900 anomalies are retained; neither difference is a common-H1-2022 intervention.',
                   interval_integral='1751..2022 (272 native one-year intervals); 1750 is an initial-state label.',
                   source_provenance=json.loads(ds.attrs['provenance_json']),
                   endpoint_2022={str(mode):{name:dist(differences.sel(driver_mode=mode)[name].sel(year=2022).values) for name in differences.data_vars if 'age_support' not in differences[name].dims} for mode in ds.driver_mode.values})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(status='complete', output=str(args.output), source=str(cache), paired_2022=summary['endpoint_2022']),ensure_ascii=False))


if __name__ == '__main__':
    main()
